//! Extension of bytes::Buf to to decode leb128 integers.
//!
//! Unlike the lucene DataInput.java implementation this won't read in an unbounded fashion if every
//! byte has the high bit set and instead return an overflow error.

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Overflow;

pub trait VIntBuf: bytes::Buf {
    #[inline(always)]
    fn get_vi32(&mut self) -> Result<i32, Overflow> {
        self.get_vi64()?.try_into().map_err(|_| Overflow)
    }

    #[inline(always)]
    fn get_vi64(&mut self) -> Result<i64, Overflow> {
        let mut v = 0i64;
        for i in 0..9 {
            let b = self.get_u8();
            v |= ((b & !0x80) as i64) << (i * 7);
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        // We've consumed 63 bits up to this point. If the next byte is > 1 then it's overflowing.
        let b = self.get_u8();
        if b <= 1 {
            Ok(v | ((b as i64) << 63))
        } else {
            Err(Overflow)
        }
    }
}

impl VIntBuf for &[u8] {}

#[cfg(test)]
mod test {
    use super::VIntBuf;

    #[test]
    fn test1() {
        let mut bytes = [
            80u8, 128, 128, 178, 212, 22, 128, 12, 48, 27, 15, 0, 255, 255, 255, 255,
        ]
        .as_slice();
        assert_eq!(bytes.get_vi64(), Ok(80));
        assert_eq!(bytes.get_vi64(), Ok(0x1_6A8C_8000));
    }
}
