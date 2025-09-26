// Tests using the index returned from String.indexOf
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;


class StringIndexOf {

    public static String remove(String l, String s) {
        int i = l.IndexOf(s);
        if (i != -1) {
            return l.Substring(0, i) + l.Substring(i + s.Length);
        }
        return l;
    }

    public static String nocheck(String l, String s) {
        int i = l.IndexOf(s);
        // :: error: (argument.type.incompatible)
        return l.Substring(0, i) + l.Substring(i + s.Length);
    }

    public static String remove(String l, String s, int from, bool last) {
        int i = last ? l.LastIndexOf(s, from) : l.IndexOf(s, from);
        if (i >= 0) {
            return l.Substring(0, i) + l.Substring(i + s.Length);
        }
        return l;
    }

    public static String stringLiteral(String l) {
        int i = l.IndexOf("constant");
        if (i != -1) {
            return l.Substring(0, i) + l.Substring(i + "constant".Length);
        }
        // :: error: (argument.type.incompatible)
        return l.Substring(0, i) + l.Substring(i + "constant".Length);
    }

    public static char character(String l, char c) {
        int i = l.IndexOf(c);
        if (i > -1) {
            return l[i];
        }
        // :: error: (argument.type.incompatible)
        return l[i];
    }
}
