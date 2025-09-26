using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;
// Tests for index annotations on string methods in the annotated JDK

class StringMethods {

    void testCharAt(String s, int i) {
        // ::  error: (argument.type.incompatible)
        var _ = s[i];
        // ::  error: (argument.type.incompatible)
        char.ConvertToUtf32(s, i);

        if (i >= 0 && i < s.Length) {
            var _1 = s[i];
            char.ConvertToUtf32(s, i);
        }
    }
#if JDK
    void testCodePointBefore(String s) {
        // ::  error: (argument.type.incompatible)
        s.codePointBefore(0);

        if (s.Length > 0) {
            s.codePointBefore(s.Length);
        }
    }
#endif

    void testSubstring(String s) {
        s.Substring(0);
        s.Substring(0, 0);
        s.Substring(s.Length);
        s.Substring(s.Length, 0);
        s.Substring(0, s.Length);
        // ::  error: (argument.type.incompatible)
        s.Substring(1);
        // ::  error: (argument.type.incompatible)
        s.Substring(0, 1);
    }

    void testIndexOf(String s, char c) {
        int i = s.IndexOf(c);
        if (i != -1) {
            var _ = s[i];
        }
    }
}
