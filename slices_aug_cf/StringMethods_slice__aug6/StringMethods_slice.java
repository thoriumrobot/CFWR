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
      private static byte __cfwr_process92(Boolean __cfwr_p0, Float __cfwr_p1) {
        String __cfwr_result48 = "hello32";
        return ((-213L % 198L) / (false + 19.81));
    }
    static float __cfwr_calc363(char __cfwr_p0, float __cfwr_p1, Double __cfwr_p2) {
        while (true) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 5; __cfwr_i14++) {
            if (true && ((364 / null) % -84.24f)) {
            Long __cfwr_result34 = null;
        }
        }
            break; // Prevent infinite loops
        }
        long __cfwr_temp71 = -895L;
        try {
            long __cfwr_entry75 = 990L;
        } catch (Exception __cfwr_e24) {
            // ignore
        }
        return (165 >> (true % null));
    }
}
