/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testCharAt(String s, int i) {
        return null;

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
      protected boolean __cfwr_temp50(long __cfwr_p0, float __cfwr_p1) {
        while ((null * false)) {
            while (true) {
            try {
            if ((('Q' ^ 'N') >> null) && (-11.08f >> (-64.21 | '0'))) {
            if (false && true) {
            try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return false;
    }
    public Long __cfwr_util222() {
        try {
            return null;
        } catch (Exception __cfwr_e3) {
            // ignore
        }
        while ((false / (null + true))) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 2; __cfwr_i11++) {
            if (false || true) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        try {
            while (true) {
            return 555;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e35) {
            // ignore
        }
        return null;
    }
}
