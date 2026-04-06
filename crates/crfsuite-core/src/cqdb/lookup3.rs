/// Bob Jenkins' lookup3 hash function (hashlittle).
/// Produces identical output to the C implementation in lookup3.c.

#[inline]
fn rot(x: u32, k: u32) -> u32 {
    (x << k) | (x >> (32 - k))
}

macro_rules! mix {
    ($a:expr, $b:expr, $c:expr) => {{
        $a = $a.wrapping_sub($c); $a ^= rot($c, 4);  $c = $c.wrapping_add($b);
        $b = $b.wrapping_sub($a); $b ^= rot($a, 6);  $a = $a.wrapping_add($c);
        $c = $c.wrapping_sub($b); $c ^= rot($b, 8);  $b = $b.wrapping_add($a);
        $a = $a.wrapping_sub($c); $a ^= rot($c, 16); $c = $c.wrapping_add($b);
        $b = $b.wrapping_sub($a); $b ^= rot($a, 19); $a = $a.wrapping_add($c);
        $c = $c.wrapping_sub($b); $c ^= rot($b, 4);  $b = $b.wrapping_add($a);
    }};
}

macro_rules! final_mix {
    ($a:expr, $b:expr, $c:expr) => {{
        $c ^= $b; $c = $c.wrapping_sub(rot($b, 14));
        $a ^= $c; $a = $a.wrapping_sub(rot($c, 11));
        $b ^= $a; $b = $b.wrapping_sub(rot($a, 25));
        $c ^= $b; $c = $c.wrapping_sub(rot($b, 16));
        $a ^= $c; $a = $a.wrapping_sub(rot($c, 4));
        $b ^= $a; $b = $b.wrapping_sub(rot($a, 14));
        $c ^= $b; $c = $c.wrapping_sub(rot($b, 24));
    }};
}

/// Hash a byte slice. This is the "unaligned/big-endian" path from lookup3.c
/// which works correctly on all platforms.
pub fn hashlittle(key: &[u8], initval: u32) -> u32 {
    let length = key.len();
    let mut a: u32 = 0xdeadbeef_u32
        .wrapping_add(length as u32)
        .wrapping_add(initval);
    let mut b = a;
    let mut c = a;

    let mut offset = 0;
    let mut remaining = length;

    // Process 12 bytes at a time
    while remaining > 12 {
        let k = &key[offset..];
        a = a.wrapping_add(
            k[0] as u32
                | (k[1] as u32) << 8
                | (k[2] as u32) << 16
                | (k[3] as u32) << 24,
        );
        b = b.wrapping_add(
            k[4] as u32
                | (k[5] as u32) << 8
                | (k[6] as u32) << 16
                | (k[7] as u32) << 24,
        );
        c = c.wrapping_add(
            k[8] as u32
                | (k[9] as u32) << 8
                | (k[10] as u32) << 16
                | (k[11] as u32) << 24,
        );
        mix!(a, b, c);
        offset += 12;
        remaining -= 12;
    }

    // Handle the last few bytes (the switch/case in C)
    let k = &key[offset..];
    match remaining {
        12 => { c = c.wrapping_add((k[11] as u32) << 24);
                c = c.wrapping_add((k[10] as u32) << 16);
                c = c.wrapping_add((k[9] as u32) << 8);
                c = c.wrapping_add(k[8] as u32);
                b = b.wrapping_add((k[7] as u32) << 24);
                b = b.wrapping_add((k[6] as u32) << 16);
                b = b.wrapping_add((k[5] as u32) << 8);
                b = b.wrapping_add(k[4] as u32);
                a = a.wrapping_add((k[3] as u32) << 24);
                a = a.wrapping_add((k[2] as u32) << 16);
                a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        11 => { c = c.wrapping_add((k[10] as u32) << 16);
                c = c.wrapping_add((k[9] as u32) << 8);
                c = c.wrapping_add(k[8] as u32);
                b = b.wrapping_add((k[7] as u32) << 24);
                b = b.wrapping_add((k[6] as u32) << 16);
                b = b.wrapping_add((k[5] as u32) << 8);
                b = b.wrapping_add(k[4] as u32);
                a = a.wrapping_add((k[3] as u32) << 24);
                a = a.wrapping_add((k[2] as u32) << 16);
                a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        10 => { c = c.wrapping_add((k[9] as u32) << 8);
                c = c.wrapping_add(k[8] as u32);
                b = b.wrapping_add((k[7] as u32) << 24);
                b = b.wrapping_add((k[6] as u32) << 16);
                b = b.wrapping_add((k[5] as u32) << 8);
                b = b.wrapping_add(k[4] as u32);
                a = a.wrapping_add((k[3] as u32) << 24);
                a = a.wrapping_add((k[2] as u32) << 16);
                a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        9 =>  { c = c.wrapping_add(k[8] as u32);
                b = b.wrapping_add((k[7] as u32) << 24);
                b = b.wrapping_add((k[6] as u32) << 16);
                b = b.wrapping_add((k[5] as u32) << 8);
                b = b.wrapping_add(k[4] as u32);
                a = a.wrapping_add((k[3] as u32) << 24);
                a = a.wrapping_add((k[2] as u32) << 16);
                a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        8 =>  { b = b.wrapping_add((k[7] as u32) << 24);
                b = b.wrapping_add((k[6] as u32) << 16);
                b = b.wrapping_add((k[5] as u32) << 8);
                b = b.wrapping_add(k[4] as u32);
                a = a.wrapping_add((k[3] as u32) << 24);
                a = a.wrapping_add((k[2] as u32) << 16);
                a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        7 =>  { b = b.wrapping_add((k[6] as u32) << 16);
                b = b.wrapping_add((k[5] as u32) << 8);
                b = b.wrapping_add(k[4] as u32);
                a = a.wrapping_add((k[3] as u32) << 24);
                a = a.wrapping_add((k[2] as u32) << 16);
                a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        6 =>  { b = b.wrapping_add((k[5] as u32) << 8);
                b = b.wrapping_add(k[4] as u32);
                a = a.wrapping_add((k[3] as u32) << 24);
                a = a.wrapping_add((k[2] as u32) << 16);
                a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        5 =>  { b = b.wrapping_add(k[4] as u32);
                a = a.wrapping_add((k[3] as u32) << 24);
                a = a.wrapping_add((k[2] as u32) << 16);
                a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        4 =>  { a = a.wrapping_add((k[3] as u32) << 24);
                a = a.wrapping_add((k[2] as u32) << 16);
                a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        3 =>  { a = a.wrapping_add((k[2] as u32) << 16);
                a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        2 =>  { a = a.wrapping_add((k[1] as u32) << 8);
                a = a.wrapping_add(k[0] as u32);
        }
        1 =>  { a = a.wrapping_add(k[0] as u32);
        }
        0 =>  { return c; } // nothing to add
        _ => unreachable!(),
    }
    final_mix!(a, b, c);
    c
}

/// Hash a string the way CQDB does: including the null terminator.
pub fn hash_string(s: &str) -> u32 {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0); // null terminator
    hashlittle(&bytes, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_with_null() {
        // Hash of just a null byte (empty C string)
        let result = hashlittle(&[0], 0);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_known_strings() {
        // These can be verified against the C implementation
        let h1 = hash_string("B-NP");
        let h2 = hash_string("I-NP");
        let h3 = hash_string("B-VP");
        // Just check they're different and non-zero
        assert_ne!(h1, 0);
        assert_ne!(h2, 0);
        assert_ne!(h3, 0);
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h2, h3);
    }
}
