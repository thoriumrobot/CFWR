using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;
/*@SuppressWarnings("array.access.unsafe.high")*/
public class HexEncode {
    private static readonly char[] digits = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
    };

    public static String hexEncode(byte[] bytes) {
        StringBuilder s = new StringBuilder(bytes.Length * 2);
        for (int i = 0; i < bytes.Length; i++) {
            byte b = bytes[i];
            s.Append(digits[(b & 0xf0) >> 4]);
            s.Append(digits[b & 0x0f]);
        }
        return s.ToString();
    }
}
