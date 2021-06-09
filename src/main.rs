use std::{
    fmt::{Debug, Display},
    fs::File,
    io::{stdout, Read, Stdout, Write},
    ops::{Index, IndexMut},
    process::exit,
    sync::{
        atomic::{AtomicU16, AtomicU8, Ordering},
        mpsc::{self, Receiver, Sender, TryRecvError},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use argh::FromArgs;
use crossterm::{
    cursor::{self, MoveTo},
    event::{self, KeyCode, KeyModifiers},
    style::{Color, Colors, Print, ResetColor, SetBackgroundColor, SetColors},
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
    QueueableCommand,
};
use rand::{rngs::ThreadRng, thread_rng, Rng};

type C8Result<T> = Result<T, String>;
trait IntoC8Result<T, M: Display> {
    fn c8_err(self, msg: M) -> C8Result<T>;
}
impl<T, E: Debug, M: Display> IntoC8Result<T, M> for Result<T, E> {
    fn c8_err(self, msg: M) -> C8Result<T> {
        self.map_err(|e| format!("{}\nOriginal error:\n{:?}", msg, e))
    }
}

type Nibble = u8;
type Addr = u16;

const MEMORY_SIZE: usize = 0x1000;
const FONT_START: usize = 0x000;
const FONT_SIZE: usize = 5;
const PROGRAM_START: usize = 0x200;
const STACK_DEPTH: usize = 32;
const SPRITE_WIDTH: u8 = 8;
const SCREEN_HEIGHT: usize = 32;
const SCREEN_HEIGHT_TERMINAL_UNITS: usize = 16;
const SCREEN_WIDTH: usize = 64;
const SCREEN_OFFSET_X: u16 = 0;
const SCREEN_OFFSET_Y: u16 = 0;
const PIXEL_EMPTY: &str = " ";
// unicode bottom-half box character
const PIXEL_HALF: &str = "\u{2584}";
const SCREEN_ON_COLOR: Color = Color::White;
const SCREEN_OFF_COLOR: Color = Color::DarkGrey;
const TIMER_TICK_RATE_HZ: f32 = 60.;
const INPUT_TIMEOUT: Duration = Duration::from_millis(17);

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
macro_rules! INPUT_MATCH {
    ($target:expr => $result:expr) => {
        INPUT_MATCH!(@inner $target; $result; 'x'0'1'1'2'2'3'3'q'4'w'5'e'6'a'7's'8'd'9'z'10'c'11'4'12'r'13'f'14'v'15);
    };
    (@inner $target:expr; $result:expr; $($chr:literal $idx:literal)*) => {
        match ($target).code {
            $(
                KeyCode::Char($chr) => {($result)($idx)},
            )*
            _ => ()
        }
    };
}

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
    data: [u8; MEMORY_SIZE],
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
    fn from_rom(rom: &[u8]) -> C8Result<Self> {
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
    fn get<N: Into<usize>>(&self, addr: N) -> &u8 {
        &self.data[addr.into()]
    }
    fn get_offset<N: Into<usize>>(&self, offset: N) -> &u8 {
        &self.data[self.ptr as usize + offset.into()]
    }
    fn get_offset_mut<N: Into<usize>>(&mut self, offset: N) -> &mut u8 {
        &mut self.data[self.ptr as usize + offset.into()]
    }
    fn get_slice<N: Into<usize>>(&self, length: N) -> &[u8] {
        &self.data[self.ptr as usize..self.ptr as usize + length.into()]
    }
}

/// Data registers holding bytes.
///
/// The data registers V0-VF are one byte in size.
/// VF is also used for the carry flag for addition,
/// the no-borrow flag for subtraction, and pixel collision for sprite drawing.
#[derive(Debug, Default)]
struct Registers([u8; 16]);

type Reg = Nibble;
const V0: Reg = 0x0;
const VF: Reg = 0xF;

impl Index<Reg> for Registers {
    type Output = u8;

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
        self[VF] = flag as u8;
    }
}

/// The call stack.
#[derive(Debug, Default)]
struct Stack {
    data: [Addr; STACK_DEPTH],
    ptr: u8,
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
struct Framebuffer {
    data: [u64; SCREEN_HEIGHT],
    stream: Stdout,
}

impl Default for Framebuffer {
    fn default() -> Self {
        Self {
            data: [0; SCREEN_HEIGHT],
            stream: stdout(),
        }
    }
}

impl Framebuffer {
    fn draw_slice(&mut self, top: u8, bottom: u8, x: u8) {
        for i in 0..(SCREEN_WIDTH as u8 - x).min(SPRITE_WIDTH) {
            let top_px = top & 1 << i != 0;
            let bottom_px = bottom & 1 << i != 0;
            if top_px ^ bottom_px {
                self.stream
                    .queue(SetColors(Colors::new(
                        if bottom_px {
                            SCREEN_ON_COLOR
                        } else {
                            SCREEN_OFF_COLOR
                        },
                        if top_px {
                            SCREEN_ON_COLOR
                        } else {
                            SCREEN_OFF_COLOR
                        },
                    )))
                    .unwrap();
                self.stream.queue(Print(PIXEL_HALF)).unwrap();
            } else {
                self.stream
                    .queue(SetBackgroundColor(if top_px {
                        SCREEN_ON_COLOR
                    } else {
                        SCREEN_OFF_COLOR
                    }))
                    .unwrap();
                self.stream.queue(Print(PIXEL_EMPTY)).unwrap();
            }
        }
    }
    fn move_to(&mut self, x: u8, y: u8) {
        self.stream
            .queue(MoveTo(
                SCREEN_OFFSET_X + x as u16,
                SCREEN_OFFSET_Y + y as u16 / 2,
            ))
            .unwrap();
    }
    fn udpate(&mut self, slice: u8, x: u8, y: u8) -> (bool, u8) {
        let orig = self.data[y as usize];
        let shifted = (slice as u64) << x as u64;
        let result = orig ^ shifted;
        self.data[y as usize] = result;
        (orig & !result != 0, (result >> x as u64) as u8)
    }
    fn draw_sprite(&mut self, x: u8, y: u8, sprite: &[u8]) -> bool {
        // todo rewrite the whole thing
        // handle x > off limits
        let x_pos = x % SCREEN_WIDTH as u8;
        let y_pos = y % SCREEN_HEIGHT as u8;
        let mut flag = false;
        let align_offset = y_pos & 1;
        if align_offset != 0 {
            let top = (self.data[y_pos as usize - 1] >> x_pos as u64) as u8;
            let (f, bottom) = self.udpate(sprite[0].reverse_bits(), x_pos, y_pos);
            flag |= f;
            self.move_to(x_pos, y_pos - 1);
            self.draw_slice(top, bottom, x_pos);
        }
        for (i, pair) in sprite[align_offset as usize..].chunks(2).enumerate() {
            if let &[top, bottom] = pair {
                let y_offset = y_pos + align_offset + 2 * i as u8;
                let (f_t, top) = self.udpate(top.reverse_bits(), x_pos, y_offset);
                let (f_b, bottom) = self.udpate(bottom.reverse_bits(), x_pos, y_offset + 1);
                flag |= f_t | f_b;
                self.move_to(x_pos, y_offset);
                self.draw_slice(top, bottom, x_pos);
            } else if let &[top] = pair {
                let y_offset = y_pos + align_offset + 2 * i as u8;
                let (f, top) = self.udpate(top.reverse_bits(), x_pos, y_offset);
                flag |= f;
                let bottom = (self.data[y_offset as usize + 1] >> x_pos as u64) as u8;
                self.move_to(x_pos, y_offset);
                self.draw_slice(top, bottom, x_pos);
            }
        }
        self.stream.flush().unwrap();
        flag
    }
    fn clear(&mut self) {
        self.data = [0; SCREEN_HEIGHT];
        for y in 0..SCREEN_HEIGHT_TERMINAL_UNITS {
            self.stream
                .queue(MoveTo(SCREEN_OFFSET_X, SCREEN_OFFSET_Y + y as u16))
                .unwrap();
            self.stream
                .queue(SetBackgroundColor(SCREEN_OFF_COLOR))
                .unwrap();
            let empty_row = PIXEL_EMPTY.repeat(SCREEN_WIDTH);
            self.stream.queue(Print(empty_row)).unwrap();
        }
        self.stream.flush().unwrap();
    }
}

#[derive(Debug, Default)]
struct Input {
    bits: Arc<AtomicU16>,
}

impl Input {
    fn is_pressed(&self, x: Nibble) -> bool {
        self.bits.load(Ordering::SeqCst) & (1 << x) != 0
    }
    fn try_get_key(&self) -> Option<Nibble> {
        let bits = self.bits.load(Ordering::SeqCst);
        if bits != 0 {
            Some(bits.trailing_zeros() as Nibble)
        } else {
            None
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
fn byte(b_0: u8, b_1: u8) -> u8 {
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
    fn se_vx_bb(&mut self, x: Reg, b: u8) {
        if self.reg[x] == b {
            self.pc += 2;
        }
    }
    fn sne_vx_bb(&mut self, x: Reg, b: u8) {
        if self.reg[x] != b {
            self.pc += 2;
        }
    }
    fn se_vx_vy(&mut self, x: Reg, y: Reg) {
        if self.reg[x] == self.reg[y] {
            self.pc += 2;
        }
    }
    fn ld_vx_bb(&mut self, x: Reg, b: u8) {
        self.reg[x] = b;
    }
    fn add_vx_bb(&mut self, x: Reg, b: u8) {
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
        let ret = self.reg[x] + self.reg[y];
        self.reg.set_flag(ret < self.reg[x]);
        self.reg[x] = ret;
    }
    fn sub_vx_vy(&mut self, x: Reg, y: Reg) {
        let ret = self.reg[x] - self.reg[y];
        self.reg.set_flag(self.reg[x] > self.reg[y]);
        self.reg[x] = ret;
    }
    fn shr_vx_vy(&mut self, x: Reg, _y: Reg) {
        self.reg.set_flag(self.reg[x] & 1 == 1);
        self.reg[x] >>= 1;
    }
    fn subn_vx_vy(&mut self, x: Reg, y: Reg) {
        let ret = self.reg[y] - self.reg[x];
        self.reg.set_flag(self.reg[y] > self.reg[x]);
        self.reg[x] = ret;
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
    fn rnd_vx_bb(&mut self, x: Reg, b: u8) {
        self.reg[x] = self.rng.gen::<u8>() & b;
    }
    fn drw_vx_vy_n(&mut self, x: Reg, y: Reg, n: Nibble) {
        let flag = self.buf.draw_sprite(
            self.reg[x] & (SCREEN_WIDTH as u8 - 1),
            self.reg[y] & (SCREEN_HEIGHT as u8 - 1),
            self.mem.get_slice(n),
        );
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
        self.reg[x] = self.timers.dt.load(Ordering::SeqCst);
    }
    fn ld_vx_k(&mut self, x: Reg) {
        if let Some(key) = self.input.try_get_key() {
            self.reg[x] = key;
        } else {
            self.pc -= 2;
        }
    }
    fn ld_dt_vx(&mut self, x: Reg) {
        self.timers.dt.store(self.reg[x], Ordering::SeqCst);
    }
    fn ld_st_vx(&mut self, x: Reg) {
        self.timers.st.store(self.reg[x], Ordering::SeqCst);
    }
    fn add_i_vx(&mut self, x: Reg) {
        self.mem.ptr += self.reg[x] as Addr;
    }
    fn ld_f_vx(&mut self, x: Reg) {
        self.mem.ptr = FONT_START as Addr + self.reg[x] as Addr * FONT_SIZE as Addr;
    }
    fn ld_b_vx(&mut self, x: Reg) {
        let val = self.reg[x];
        *self.mem.get_offset_mut(0u8) = val / 100;
        *self.mem.get_offset_mut(1u8) = val / 10 % 10;
        *self.mem.get_offset_mut(2u8) = val % 10;
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

enum HaltSignal {
    Halt,
}
enum InputSignal {
    Exit,
}
enum ExitKind {
    Forced,
    Natural,
}

struct ChannelHandler {
    timer_tx: Sender<HaltSignal>,
    input_tx: Sender<HaltSignal>,
    input_rx: Receiver<InputSignal>,
}

impl ChannelHandler {
    fn halt(&self) -> C8Result<()> {
        for tx in &[&self.input_tx, &self.timer_tx] {
            tx.send(HaltSignal::Halt)
                .c8_err("Could not send message to thread")?;
        }
        Ok(())
    }
}

impl Chip8Machine {
    fn new(rom: &[u8]) -> C8Result<Self> {
        Ok(Self {
            mem: Memory::from_rom(rom)?,
            rng: thread_rng(),
            pc: PROGRAM_START as Addr,
            ..Default::default()
        })
    }
    fn run_from_args(args: Args) -> C8Result<ExitKind> {
        let executor = if args.hex {
            Self::run_hex
        } else {
            Self::run_binary
        };
        let mut buf = vec![];
        File::open(args.file)
            .c8_err("Error occurred trying to open the file")?
            .read_to_end(&mut buf)
            .c8_err("Error occurred trying to read the file")?;
        executor(&buf)
    }
    fn run_hex(rom: &[u8]) -> C8Result<ExitKind> {
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
    fn run_binary(rom: &[u8]) -> C8Result<ExitKind> {
        let mut machine = Self::new(rom)?;
        machine.run()
    }
    fn setup_terminal(&self) -> C8Result<()> {
        let mut stdout = stdout();
        stdout
            .queue(cursor::Hide)
            .c8_err("Could not hide cursor")?
            .queue(EnterAlternateScreen)
            .c8_err("Could not enter alternate screen")?
            .flush()
            .c8_err("Could not flush stdout")?;

        terminal::enable_raw_mode().c8_err("Could not enable terminal raw mode")?;

        Ok(())
    }
    fn teardown_terminal(&self) -> C8Result<()> {
        terminal::disable_raw_mode().c8_err("Could not disable terminal raw mode")?;
        let mut stdout = stdout();
        stdout
            .queue(LeaveAlternateScreen)
            .c8_err("Could not leave alternate screen")?
            .queue(cursor::Show)
            .c8_err("Could not show cursor")?
            .queue(ResetColor)
            .c8_err("Could not reset colors")?
            .flush()
            .c8_err("Could not flush stdout")?;

        Ok(())
    }
    fn spawn_extra_threads(&self) -> C8Result<ChannelHandler> {
        let timer_tx = self.spawn_timer()?;
        let (display_rx, input_tx) = self.spawn_display()?;
        Ok(ChannelHandler {
            timer_tx,
            input_tx,
            input_rx: display_rx,
        })
    }
    fn spawn_timer(&self) -> C8Result<Sender<HaltSignal>> {
        let dt = Arc::clone(&self.timers.dt);
        let st = Arc::clone(&self.timers.st);
        let (timer_tx, timer_rx) = mpsc::channel();
        thread::spawn(move || loop {
            match timer_rx.try_recv() {
                Ok(HaltSignal::Halt) | Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => (),
            }
            // Result is ignored, since we don't care about the
            // registers when they're zero
            let _ = dt.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                if n != 0 {
                    Some(n - 1)
                } else {
                    None
                }
            });
            let _ = st.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                if n != 0 {
                    Some(n - 1)
                } else {
                    None
                }
            });
            thread::sleep(Duration::from_secs_f32(1. / TIMER_TICK_RATE_HZ));
        });
        Ok(timer_tx)
    }
    fn spawn_display(&self) -> C8Result<(Receiver<InputSignal>, Sender<HaltSignal>)> {
        let input = Arc::clone(&self.input.bits);
        let (display_tx, display_rx) = mpsc::channel();
        let (input_tx, input_rx) = mpsc::channel();
        thread::spawn(move || {
            loop {
                let mut starts: [Option<Instant>; 16] = [None; 16];
                match input_rx.try_recv() {
                    Ok(HaltSignal::Halt) | Err(TryRecvError::Disconnected) => break,
                    Err(TryRecvError::Empty) => (),
                }
                let next_timeout = starts
                    .iter()
                    .flatten()
                    .min()
                    .map_or(INPUT_TIMEOUT, |inst| inst.duration_since(Instant::now()));
                let result = event::poll(next_timeout);
                starts = {
                    let mut new = [None; 16];
                    for (i, &start) in starts.iter().enumerate() {
                        new[i] = if let Some(inst) = start {
                            if inst <= Instant::now() {
                                None
                            } else {
                                Some(inst)
                            }
                        } else {
                            None
                        }
                    }
                    new
                };
                if let Ok(true) = result {
                    if let Ok(event::Event::Key(k)) = event::read() {
                        if let KeyModifiers::NONE = k.modifiers {
                            INPUT_MATCH! {
                                k => |i| {starts[i] = Some(Instant::now() + INPUT_TIMEOUT)}
                            }
                        } else {
                            match k.code {
                                KeyCode::Char('c') => {
                                    display_tx.send(InputSignal::Exit).unwrap();
                                    // prevents race conditions as the thread can be killed at any point after this line
                                    thread::sleep(Duration::from_secs(300));
                                }
                                _ => (),
                            }
                        }
                    }
                }
                let mut key_state = 0u16;
                for i in 0..16 {
                    key_state |= (starts[i].is_some() as u16) << i;
                }
                input.store(key_state, Ordering::SeqCst);
            }
        });
        Ok((display_rx, input_tx))
    }
    fn run(&mut self) -> C8Result<ExitKind> {
        self.setup_terminal()?;
        let channels = self.spawn_extra_threads()?;
        self.buf.clear();
        loop {
            match channels.input_rx.try_recv() {
                Ok(InputSignal::Exit) => {
                    self.teardown_terminal()?;
                    channels.halt()?;
                    return Ok(ExitKind::Forced);
                }
                _ => (),
            }
            if self.pc >= MEMORY_SIZE as Addr {
                self.teardown_terminal()?;
                channels.halt()?;
                return Ok(ExitKind::Natural);
            }
            let high = *self.mem.get(self.pc);
            let low = *self.mem.get(self.pc + 1);
            self.pc += 2;
            self.execute(high, low);
        }
    }
    fn execute(&mut self, high: u8, low: u8) {
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

/// Runs a chip-8 program
#[derive(FromArgs)]
struct Args {
    /// parse hex file into binary
    #[argh(switch, short = 'x')]
    hex: bool,
    /// the file to execute
    #[argh(positional)]
    file: String,
}

fn main() {
    let args: Args = argh::from_env();
    match Chip8Machine::run_from_args(args) {
        Ok(kind) => match kind {
            ExitKind::Natural => {
                exit(0);
            }
            ExitKind::Forced => {
                exit(130);
            }
        },
        Err(e) => {
            println!("An unexpected error occurred: {}", e);
            exit(1);
        }
    }
}
