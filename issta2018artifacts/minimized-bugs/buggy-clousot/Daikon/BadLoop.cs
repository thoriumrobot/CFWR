using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

// simplified version of buggy Daikon code fixed in 5a599e7a1
// a valid bug, but not out-of-bounds access
#if false
class BadLoop
{
    // Outputs a sequence of space-separated characters, with (only) return
    // and newline quoted.  (Should backslash also be quoted?)
    public static void println_array_char_as_chars(TextWriter w, object[] a)
    {
        if (a == null) {
            w.WriteLine("null");
            return;
        }
        w.Write('[');
        for (int i = 0; i < a.Length; i++)
        {
            if (i != 0) w.Write(' ');
            //:: error: array.access.unsafe.high.constant
            char c = (char)a[0];
            if (c == '\r') // not lineSep
            {
                w.Write("\\r"); // not lineSep
            }
            else if (c == '\n')
            {
                w.Write("\\n");
            }
            else
            {
                w.Write(c);
            }
        }
        w.WriteLine(']');
    }
}
#endif