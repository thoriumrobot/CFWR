/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringMethods_slice {
  void testCharAt(String s, int i) {
        for (int __cfwr_i27 = 0; __cfwr_i27 < 5; __cfwr_i27++) {
            return false;
        }

    // ::  error: (argument)
    s.charAt(i);
    // ::  error: (argument)
    s.codePointAt(i);

    if (i >= 0 && i < s.length()) {
      s.charAt(i);
      s.codePointAt(i);
    }
  }

  void testCodePointBefore(String s) {
    // ::  error: (argument)
    s.codePointBefore(0);

    if (s.length() > 0) {
      s.codePointBefore(s.length());
    }
  }

  void testSubstring(String s) {
    s.substring(0);
    s.substring(0, 0);
    s.substring(s.length());
    s.substring(s.length(), s.length());
    s.substring(0, s.length());
    // ::  error: (argument)
    s.substring(1);
    // ::  error: (argument)
    s.substring(0, 1);
  }

    private Integer __cfwr_process796() {
        if (((761L & -52.77) / -596) && true) {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 9; __cfwr_i30++) {
            if (false && true) {
            float __cfwr_var26 = 47.73f;
        }
        }
        }
        return null;
        return null;
        return null;
    }
    boolean __cfwr_proc942(String __cfwr_p0, Long __cfwr_p1) {
        char __cfwr_var47 = (421 & (8.40 / 29.75f));
        while (false) {
            while (true) {
            for (int __cfwr_i16 = 0; __cfwr_i16 < 1; __cfwr_i16++) {
            try {
            while (('O' % 943L)) {
            if ((null + ('J' - '7')) && ('U' - false)) {
            if (false && false) {
            if (false || false) {
            Float __cfwr_item93 = null;
        }
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        try {
            Object __cfwr_result11 = null;
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        return true;
    }
}