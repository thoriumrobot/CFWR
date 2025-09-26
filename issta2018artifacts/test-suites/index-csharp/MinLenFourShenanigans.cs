using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class MinLenFourShenanigans {
    public static bool isInterned(Object value) {
        if (value == null) {
            // nothing to do
            return true;
        } else if (value is String) {
            // Used to issue the below error.
            // MinLenFourShenanigans.java:7: warning: [cast.unsafe] "@MinLen(0) Object" may not be
            // casted to the type "@MinLen(4) String"
            return (value == string.Intern(((String) value)));
        }
        return false;
    }

    public static bool isInterned2(Object value) {
        if (value is String) {
            return (value == string.Intern((String) value));
        }
        return false;
    }
}
