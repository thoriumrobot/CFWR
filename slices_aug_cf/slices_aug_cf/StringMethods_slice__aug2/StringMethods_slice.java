/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testCharAt(String s, int i) {
        double __cfwr_val43 = 98.35;

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
      public String __cfwr_proc379(Float __cfwr_p0, byte __cfwr_p1, int __cfwr_p2) {
        while (false) {
            try {
            Object __cfwr_entry35 = null;
        } catch (Exception __cfwr_e51) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return "value40";
    }
    double __cfwr_func475() {
        return true;
        boolean __cfwr_node65 = (-33.66 % true);
        while (true) {
            try {
            return true;
        } catch (Exception __cfwr_e26) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
        return -46.40;
    }
}
