using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class StringSameLen {
    public void m(String s) {
        String t = s;

        for (int i = 0; i < s.Length; ++i) {
            char c = t[i];
        }
    }

    public void m2(String s) {
        String t = s.ToString();

        for (int i = 0; i < s.Length; ++i) {
            char c = t[i];
        }
    }

    public void m4(String s) {
        char[] t = s.ToCharArray();

        for (int i = 0; i < s.Length; ++i) {
            char c = t[i];
        }
    }
#if JDK
    public void m6(char[] s) {
        String t = String.valueOf(s);

        for (int i = 0; i < s.Length; ++i) {
            char c = t[i];
        }
    }

    public void m7(char[] s) {
        String t = String.copyValueOf(s);

        for (int i = 0; i < s.Length; ++i) {
            char c = t[i];
        }
    }
#endif

    public void m8(String s) {
        String t = string.Intern(s);

        for (int i = 0; i < s.Length; ++i) {
            char c = t[i];
        }
    }

    public void constructor(String s) {
        String t = new String(new char[] {'a'});
    }
}
