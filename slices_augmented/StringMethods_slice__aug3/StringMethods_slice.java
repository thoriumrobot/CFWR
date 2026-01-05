/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringMethods_slice {
    @Positive
  void testCharAt(String s, int i) {
        for (int __cfwr_i57 = 0; __cfwr_i57 < 2; __cfwr_i57++) {
            return null;
        }

    // ::  error: (argument)
    @Positive
    s.charAt(i);
    // ::  error: (argument)
    @Positive
    s.codePointAt(i);

    @Positive
    if (i >= 0 && i < s.length()) {
    @Positive
      s.charAt(i);
    @Positive
      s.codePointAt(i);
    @Positive
    }
    @Positive
  }

    @Positive
  void testCodePointBefore(String s) {
    // ::  error: (argument)
    @Positive
    s.codePointBefore(0);

    @Positive
    if (s.length() > 0) {
    @Positive
      s.codePointBefore(s.length());
    @Positive
    }
    @Positive
  }

    static byte __cfwr_proc396(Character __cfwr_p0, Float __cfwr_p1, Double __cfwr_p2) {
        try {
            if ((null + (null << null)) || (-77 & ('J' % 'O'))) {
            try {
            try {
            while (true) {
            long __cfwr_entry35 = (405 << '1');
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        } catch (Exception __cfwr_e84) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        for (int __cfwr_i96 = 0; __cfwr_i96 < 9; __cfwr_i96++) {
            try {
            try {
            if ((null - ('o' ^ 45.35f)) || false) {
            while (true) {
            return "hello76";
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        }
        return null;
    }
    Object __cfwr_process899(byte __cfwr_p0, byte __cfwr_p1) {
        return null;
        return null;
    }
}