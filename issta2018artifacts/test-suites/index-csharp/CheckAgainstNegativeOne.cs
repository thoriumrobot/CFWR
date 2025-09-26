using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class CheckAgainstNegativeOne {

    public static String replaceString(String target, String oldStr, String newStr) {
        if (oldStr.Equals("")) {
            throw new ArgumentException();
        }

        StringBuilder result = new StringBuilder();
        int lastend = 0;
        if(TestHelper.nondet()) Contract.Assert(lastend >= 0 && lastend <= target.Length);
        int pos;
        while ((pos = target.IndexOf(oldStr, lastend)) != -1) {
            result.Append(target.Substring(lastend, pos - lastend));
            result.Append(newStr);
            lastend = pos + oldStr.Length;
            if(TestHelper.nondet()) Contract.Assert(lastend >= 0 && lastend <= target.Length);
        }
        result.Append(target.Substring(lastend));
        return result.ToString();
    }
}
