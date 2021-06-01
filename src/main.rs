use std::{fmt::Debug, ops::{Index, IndexMut}, sync::{
        atomic::{AtomicU16, AtomicU8, Ordering},
        mpsc::{self, TryRecvError},
        Arc,
    }, thread, time::Duration};

use crossterm::{
    event::{self, KeyCode, KeyModifiers},
    terminal,
};
use rand::{rngs::ThreadRng, thread_rng, Rng};

type Nibble = u8;
type Byte = u8;
type Addr = u16;
type Row = u64;

const MEMORY_SIZE: usize = 0x1000;
const FONT_START: usize = 0x000;
const FONT_SIZE: usize = 5;
const PROGRAM_START: usize = 0x200;
const STACK_DEPTH: usize = 32;
const FRAME_HEIGHT: usize = 32;
const DISPLAY_REFRESH_HZ: f32 = 60.;

// Typical hex keypad:
// 1 2 3 C
// 4 5 6 D
// 7 8 9 E
// A 0 B F
// QWERTY mapping:
// 1 2 3 4
// q w e r
// a s d f
// z x c v
const INPUT_MAPPING: [char; 16] = [
    'x', '1', '2', '3', 'q', 'w', 'e', 'a', 's', 'd', 'z', 'c', '4', 'r', 'f', 'v',
];

/// The main memory of the CHIP-8 virtual machine.
///
/// This exposes 4096 bytes but can theoretically handle up to 64k.
///
/// The first 0x200 bytes are reserved for fonts.
///
/// Programs typically begin execution at address 0x200.
/// ETI 660 programs begin execution at address 0x600.
#[derive(Debug)]
struct Memory {
    data: [Byte; MEMORY_SIZE],
    ptr: Addr,
}

impl Default for Memory {
    fn default() -> Self {
        Self {
            data: [0; MEMORY_SIZE],
            ptr: 0,
        }
    }
}

impl Memory {
    fn from_rom(rom: &[Byte]) -> Result<Self, String> {
        if PROGRAM_START + rom.len() >= MEMORY_SIZE {
            Err(format!(
                "Rom size {:x} is greater than the alotted memory size",
                rom.len()
            ))
        } else {
            let mut mem = [0; MEMORY_SIZE];
            mem[PROGRAM_START..PROGRAM_START + rom.len()].copy_from_slice(rom);
            Ok(Self { data: mem, ptr: 0 })
        }
    }
    fn get<N: Into<usize>>(&self, addr: N) -> &Byte {
        &self.data[addr.into()]
    }
    fn get_offset<N: Into<usize>>(&self, offset: N) -> &Byte {
        &self.data[self.ptr as usize + offset.into()]
    }
    fn get_offset_mut<N: Into<usize>>(&mut self, offset: N) -> &mut Byte {
        &mut self.data[self.ptr as usize + offset.into()]
    }
    fn get_slice<N: Into<usize>>(&self, offset: N) -> &[Byte] {
        &self.data[self.ptr as usize..self.ptr as usize + offset.into()]
    }
}

/// Data registers holding bytes.
///
/// The data registers V0-VF are one byte in size.
/// VF is also used for the carry flag for addition,
/// the no-borrow flag for subtraction, and pixel collision for sprite drawing.
#[derive(Debug, Default)]
struct Registers([Byte; 16]);

type Reg = Nibble;
const V0: Reg = 0x0;
const VF: Reg = 0xF;

impl Index<Reg> for Registers {
    type Output = Byte;

    fn index(&self, index: Reg) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl IndexMut<Reg> for Registers {
    fn index_mut(&mut self, index: Reg) -> &mut Self::Output {
        &mut self.0[index as usize]
    }
}

impl Registers {
    fn set_flag(&mut self, flag: bool) {
        self[VF] = flag as Byte;
    }
}

/// The call stack.
#[derive(Debug, Default)]
struct Stack {
    data: [Addr; STACK_DEPTH],
    ptr: Byte,
}

impl Stack {
    fn push(&mut self, addr: Addr) {
        self.ptr += 1;
        self.data[self.ptr as usize] = addr;
    }
    fn pop(&mut self) -> Addr {
        let val = self.data[self.ptr as usize];
        self.ptr -= 1;
        val
    }
}

/// The delay and sound timers. These tick down at 60Hz.
#[derive(Debug, Default)]
struct Timers {
    dt: Arc<AtomicU8>,
    st: Arc<AtomicU8>,
}

#[derive(Debug)]
struct Framebuffer([Row; FRAME_HEIGHT]);

impl Default for Framebuffer {
    fn default() -> Self {
        Self([0; FRAME_HEIGHT])
    }
}

impl Framebuffer {
    fn draw_slice(&mut self, x: Byte, y: Byte, slice: Byte) -> bool {
        let row = self.0[y as usize];
        // parentheses required  by the parser
        let shifted = (slice as Row) << x as Row;
        let result = row ^ shifted;
        let unset = row & !result;
        self.0[y as usize] = result;
        unset != 0
    }
    fn draw_sprite(&mut self, x: Byte, y: Byte, sprite: &[Byte]) -> bool {
        let mut flag = false;
        for (offset, slice) in sprite.iter().enumerate() {
            flag |= self.draw_slice(x, y + offset as Byte, *slice);
        }
        flag
    }
    fn clear(&mut self) {
        self.0 = unsafe { std::mem::zeroed() }
    }
}

#[derive(Debug, Default)]
struct Input {
    bits: Arc<AtomicU16>,
}

impl Input {
    fn is_pressed(&self, x: Nibble) -> bool {
        self.bits.load(Ordering::Relaxed) & (1 << x) != 0
    }
    fn get_key(&self) -> Nibble {
        // this blocks until a key is pressed
        loop {
            let bits = self.bits.load(Ordering::Relaxed);
            if bits != 0 {
                break bits.trailing_zeros() as Nibble;
            }
        }
    }
}

#[derive(Debug, Default)]
struct Chip8Machine {
    mem: Memory,
    reg: Registers,
    pc: Addr,
    stack: Stack,
    buf: Framebuffer,
    timers: Timers,
    rng: ThreadRng,
    input: Input,
}

/// Please ensure that inputs are nibbles.
fn byte(b_0: u8, b_1: u8) -> Byte {
    b_0 << 4 | b_1
}
/// Please ensure that inputs are nibbles.
fn addr(a_0: u8, a_1: u8, a_2: u8) -> Addr {
    (a_0 as Addr) << 8 | (a_1 as Addr) << 4 | a_2 as Addr
}

// instructions
impl Chip8Machine {
    fn cls(&mut self) {
        self.buf.clear();
    }
    fn ret(&mut self) {
        self.pc = self.stack.pop();
    }
    fn sys_aaa(&mut self, _a: Addr) {}
    fn jp_aaa(&mut self, a: Addr) {
        self.pc = a;
    }
    fn call_aaa(&mut self, a: Addr) {
        self.stack.push(self.pc);
        self.pc = a;
    }
    fn se_vx_bb(&mut self, x: Reg, b: Byte) {
        if self.reg[x] == b {
            self.pc += 2;
        }
    }
    fn sne_vx_bb(&mut self, x: Reg, b: Byte) {
        if self.reg[x] != b {
            self.pc += 2;
        }
    }
    fn se_vx_vy(&mut self, x: Reg, y: Reg) {
        if self.reg[x] == self.reg[y] {
            self.pc += 2;
        }
    }
    fn ld_vx_bb(&mut self, x: Reg, b: Byte) {
        self.reg[x] = b;
    }
    fn add_vx_bb(&mut self, x: Reg, b: Byte) {
        self.reg[x] += b;
    }
    fn ld_vx_vy(&mut self, x: Reg, y: Reg) {
        self.reg[x] = self.reg[y];
    }
    fn or_vx_vy(&mut self, x: Reg, y: Reg) {
        self.reg[x] |= self.reg[y];
    }
    fn and_vx_vy(&mut self, x: Reg, y: Reg) {
        self.reg[x] &= self.reg[y];
    }
    fn xor_vx_vy(&mut self, x: Reg, y: Reg) {
        self.reg[x] ^= self.reg[y];
    }
    fn add_vx_vy(&mut self, x: Reg, y: Reg) {
        let (ret, flag) = self.reg[x].overflowing_add(self.reg[y]);
        self.reg[x] = ret;
        self.reg.set_flag(flag);
    }
    fn sub_vx_vy(&mut self, x: Reg, y: Reg) {
        let (ret, flag) = self.reg[x].overflowing_sub(self.reg[y]);
        self.reg[x] = ret;
        self.reg.set_flag(!flag);
    }
    fn shr_vx_vy(&mut self, x: Reg, _y: Reg) {
        self.reg.set_flag(self.reg[x] & 1 == 1);
        self.reg[x] >>= 1;
    }
    fn subn_vx_vy(&mut self, x: Reg, y: Reg) {
        let (ret, flag) = self.reg[y].overflowing_sub(self.reg[x]);
        self.reg[x] = ret;
        self.reg.set_flag(!flag); // !
    }
    fn shl_vx_vy(&mut self, x: Reg, _y: Reg) {
        self.reg.set_flag(self.reg[x] & 0b10000000 == 0b10000000);
        self.reg[x] <<= 1;
    }
    fn sne_vx_vy(&mut self, x: Reg, y: Reg) {
        if self.reg[x] != self.reg[y] {
            self.pc += 2;
        }
    }
    fn ld_i_aaa(&mut self, a: Addr) {
        self.mem.ptr = a;
    }
    fn jp_v0_aaa(&mut self, a: Addr) {
        self.pc = self.reg[V0] as Addr + a;
    }
    fn rnd_vx_bb(&mut self, x: Reg, b: Byte) {
        self.reg[x] = self.rng.gen::<Byte>() & b;
    }
    fn drw_vx_vy_n(&mut self, x: Reg, y: Reg, n: Nibble) {
        let flag = self.buf.draw_sprite(x, y, self.mem.get_slice(n));
        self.reg.set_flag(flag);
    }
    fn skp_vx(&mut self, x: Reg) {
        if self.input.is_pressed(x) {
            self.pc += 2;
        }
    }
    fn sknp_vx(&mut self, x: Reg) {
        if !self.input.is_pressed(x) {
            self.pc += 2;
        }
    }
    fn ld_vx_dt(&mut self, x: Reg) {
        self.reg[x] = self.timers.dt.load(Ordering::Relaxed);
    }
    fn ld_vx_k(&mut self, x: Reg) {
        self.reg[x] = self.input.get_key();
    }
    fn ld_dt_vx(&mut self, x: Reg) {
        self.timers.dt.store(self.reg[x], Ordering::Relaxed);
    }
    fn ld_st_vx(&mut self, x: Reg) {
        self.timers.st.store(self.reg[x], Ordering::Relaxed);
    }
    fn add_i_vx(&mut self, x: Reg) {
        self.mem.ptr += self.reg[x] as Addr;
    }
    fn ld_f_vx(&mut self, x: Reg) {
        self.mem.ptr = FONT_START as Addr + self.reg[x] as Addr * FONT_SIZE as Addr;
    }
    fn ld_b_vx(&mut self, x: Reg) {
        *self.mem.get_offset_mut(0u8) = self.reg[x] / 100;
        *self.mem.get_offset_mut(1u8) = self.reg[x] % 100 / 10;
        *self.mem.get_offset_mut(2u8) = self.reg[x] % 100;
    }
    fn ld_i_vx(&mut self, x: Reg) {
        for i in 0..=x {
            *self.mem.get_offset_mut(i) = self.reg[i];
        }
    }
    fn ld_vx_i(&mut self, x: Reg) {
        for i in 0..=x {
            self.reg[i] = *self.mem.get_offset(i);
        }
    }
}
const TIMER_TERM_SIGNAL: () = ();
const DISPLAY_TERM_SIGNAL: () = ();
const DISPLAY_REFRESH_SIGNAL: bool = false;
const DISPLAY_EXIT_SIGNAL: bool = true;

enum ExitKind {
    Forced,
    Natural,
}

impl Chip8Machine {
    fn new(rom: &[Byte]) -> Result<Self, String> {
        Ok(Self {
            mem: Memory::from_rom(rom)?,
            rng: thread_rng(),
            pc: PROGRAM_START as Addr,
            ..Default::default()
        })
    }
    fn run_hex(rom: &[Byte]) -> Result<ExitKind, String> {
        let mut bin = vec![];
        let hex = |c: u8| match c {
            n @ b'0'..=b'9' => Ok(n - b'0'),
            x @ b'a'..=b'f' => Ok(x - b'a' + 10),
            x @ b'A'..=b'F' => Ok(x - b'a' + 10),
            n => Err(format!("Byte {} in input is not a hex character", n)),
        };
        if rom.len() & 1 == 1 {
            return Err(String::from("The program has an odd number of hex digits."));
        }
        for chunk in rom.chunks(2) {
            let h = chunk[0];
            let l = chunk[1];
            bin.push(hex(h)? << 4 | hex(l)?);
        }
        let mut machine = Self::new(rom)?;
        machine.run()
    }
    fn run_binary(rom: &[Byte]) -> Result<ExitKind, String> {
        let mut machine = Self::new(rom)?;
        machine.run()
    }
    fn run(&mut self) -> Result<ExitKind, String> {
        terminal::enable_raw_mode()
            .map_err(|e| format!("Could not enable terminal raw mode: {:?}", e))?;
        let dt = Arc::clone(&self.timers.dt);
        let st = Arc::clone(&self.timers.st);
        let (timer_tx, timer_rx) = mpsc::channel();
        thread::spawn(move || loop {
            match timer_rx.try_recv() {
                Ok(TIMER_TERM_SIGNAL) | Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => (),
            }
            // Result is ignored, since we don't care about the
            // registers when they're zero
            let _ = dt.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                if n != 0 {
                    Some(n - 1)
                } else {
                    None
                }
            });
            let _ = st.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                if n != 0 {
                    Some(n - 1)
                } else {
                    None
                }
            });
            thread::sleep(Duration::from_secs_f32(1. / 60.));
        });
        let input = Arc::clone(&self.input.bits);
        let (display_tx, display_rx) = mpsc::channel();
        let (input_tx, input_rx) = mpsc::channel();
        thread::spawn(move || loop {
            match input_rx.try_recv() {
                Ok(DISPLAY_TERM_SIGNAL) | Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => (),
            }
            let mut key_state = 0u16;
            while event::poll(Duration::from_millis(0)).unwrap() {
                if let event::Event::Key(k) = event::read().unwrap() {
                    match k.code {
                        KeyCode::Char('c') if k.modifiers == KeyModifiers::CONTROL => {
                            display_tx.send(DISPLAY_EXIT_SIGNAL).unwrap();
                            // prevents race conditions as the thread can be killed at any point after this line
                            thread::sleep(Duration::from_secs(300));
                        }
                        KeyCode::Char(c)
                            if k.modifiers == KeyModifiers::NONE && INPUT_MAPPING.contains(&c) =>
                        {
                            key_state |= 1 << INPUT_MAPPING.iter().position(|&x| x == c).unwrap();
                        }
                        _ => (),
                    }
                }
            }
            input.store(key_state, Ordering::Relaxed);
            display_tx.send(DISPLAY_REFRESH_SIGNAL).unwrap();
            thread::sleep(Duration::from_secs_f32(1. / DISPLAY_REFRESH_HZ));
        });
        loop {
            if self.pc >= MEMORY_SIZE as Addr {
                timer_tx.send(TIMER_TERM_SIGNAL).unwrap();
                input_tx.send(DISPLAY_TERM_SIGNAL).unwrap();
                return Ok(ExitKind::Natural);
            }
            match display_rx.try_recv() {
                Ok(DISPLAY_EXIT_SIGNAL) => {
                    timer_tx.send(TIMER_TERM_SIGNAL).unwrap();
                    input_tx.send(DISPLAY_TERM_SIGNAL).unwrap();
                    return Ok(ExitKind::Forced);
                }
                Ok(DISPLAY_REFRESH_SIGNAL) => {}
                _ => (),
            }
            let high = *self.mem.get(self.pc);
            let low = *self.mem.get(self.pc + 1);
            self.pc += 2;
            self.execute(high, low);
        }
    }
    fn execute(&mut self, high: Byte, low: Byte) {
        let h_h = high >> 4;
        let h_l = high & 0b00001111;
        let l_h = low >> 4;
        let l_l = low & 0b00001111;
        // uh oh
        match (h_h, h_l, l_h, l_l) {
            (0, 0, 0xe, 0) => self.cls(),
            (0, 0, 0xe, 0xe) => self.ret(),
            (0, a_0, a_1, a_2) => self.sys_aaa(addr(a_0, a_1, a_2)),
            (1, a_0, a_1, a_2) => self.jp_aaa(addr(a_0, a_1, a_2)),
            (2, a_0, a_1, a_2) => self.call_aaa(addr(a_0, a_1, a_2)),
            (3, x, b_0, b_1) => self.se_vx_bb(x, byte(b_0, b_1)),
            (4, x, b_0, b_1) => self.sne_vx_bb(x, byte(b_0, b_1)),
            (5, x, y, 0) => self.se_vx_vy(x, y),
            (6, x, b_0, b_1) => self.ld_vx_bb(x, byte(b_0, b_1)),
            (7, x, b_0, b_1) => self.add_vx_bb(x, byte(b_0, b_1)),
            (8, x, y, 0) => self.ld_vx_vy(x, y),
            (8, x, y, 1) => self.or_vx_vy(x, y),
            (8, x, y, 2) => self.and_vx_vy(x, y),
            (8, x, y, 3) => self.xor_vx_vy(x, y),
            (8, x, y, 4) => self.add_vx_vy(x, y),
            (8, x, y, 5) => self.sub_vx_vy(x, y),
            (8, x, y, 6) => self.shr_vx_vy(x, y),
            (8, x, y, 7) => self.subn_vx_vy(x, y),
            (8, x, y, 0xe) => self.shl_vx_vy(x, y),
            (9, x, y, 0) => self.sne_vx_vy(x, y),
            (0xa, a_0, a_1, a_2) => self.ld_i_aaa(addr(a_0, a_1, a_2)),
            (0xb, a_0, a_1, a_2) => self.jp_v0_aaa(addr(a_0, a_1, a_2)),
            (0xc, x, b_0, b_1) => self.rnd_vx_bb(x, byte(b_0, b_1)),
            (0xd, x, y, n) => self.drw_vx_vy_n(x, y, n),
            (0xe, x, 8, 0xe) => self.skp_vx(x),
            (0xe, x, 0xa, 1) => self.sknp_vx(x),
            (0xf, x, 0, 7) => self.ld_vx_dt(x),
            (0xf, x, 0, 0xa) => self.ld_vx_k(x),
            (0xf, x, 1, 5) => self.ld_dt_vx(x),
            (0xf, x, 1, 8) => self.ld_st_vx(x),
            (0xf, x, 1, 0xe) => self.add_i_vx(x),
            (0xf, x, 2, 9) => self.ld_f_vx(x),
            (0xf, x, 3, 3) => self.ld_b_vx(x),
            (0xf, x, 5, 5) => self.ld_i_vx(x),
            (0xf, x, 6, 5) => self.ld_vx_i(x),
            _ => {} // either an impossible or invalid instruction
        }
    }
}

fn main() {
    match Chip8Machine::run_binary(b"\x12\x00") {
        Ok(kind) => {
            terminal::disable_raw_mode().unwrap();
            match kind {
                ExitKind::Natural => {
                    println!("Program exited gracefully");
                }
                ExitKind::Forced => {
                    println!("Program killed");
                }
            }
        }
        Err(e) => panic!("{}", e),
    }
}
