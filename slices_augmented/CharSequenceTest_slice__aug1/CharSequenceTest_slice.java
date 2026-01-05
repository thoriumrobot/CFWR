/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CharSequenceTest_slice {
    @Positive
  void testSubSequence() {
        while (false) {
            return null;
            break; // Prevent infinite loops
        }

    // Local variable used because of https://github.com/kelloggm/checker-framework/issues/165
    @Positive
    String str = "0123456789";
    @Positive
    str.subSequence(5, 8);
    // :: error: (argument)
    @Positive
    str.subSequence(5, 13);
    @Positive
  }

  // Dummy method that takes a CharSequence and its index
    @Positive
  void sink(CharSequence cs, @IndexOrHigh("#1") int i) {}

  // Tests passing sequences as CharSequence
    @Positive
  void argumentPassing() {
    @Positive
    String s = "0123456789";
    @Positive
    sink(s, 8);
    @Positive
    StringBuilder sb = new StringBuilder("0123456789");
    // :: error: (argument)
    @Positive
    sink(sb, 8);
    @Positive
  }

  // Tests forwardning sequences as CharSequence
    @Positive
  void agumentForwarding(String s, @IndexOrHigh("#1") int i) {
    @Positive
    sink(s, i);
    @Positive
  }

  // Tests concatenation of CharSequence and String
    @Positive
  void concat() {
    @Positive
    CharSequence a = "a";
    @Positive
    @StringVal({"nullb", "ab"}) CharSequence ab = a + "b";
    @Positive
    sink(ab, 2);
    @Positive
  }

  // Tests that length retrieved from CharSequence can be used as an index
    @Positive
  void getLength(CharSequence cs, int i) {
    @Positive
    if (i >= 0 && i < cs.length()) {
    @Positive
      cs.charAt(i);
    @Positive
    }

    @Positive
    @IndexOrHigh("cs") int l = cs.length();
    @Positive
  }

    protected Integer __cfwr_helper514(String __cfwr_p0) {
        if (((94.62f << -29L) & (557L / null)) && (54.14 << null)) {
            while (false) {
            Character __cfwr_val62 = null;
            break; // Prevent infinite loops
        }
        }
        try {
            while (false) {
            while ((-20.26 ^ -52.61f)) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 10; __cfwr_i6++) {
            for (int __cfwr_i6 = 0; __cfwr_i6 < 3; __cfwr_i6++) {
            while ((null - 19.24)) {
            return null;
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        while ((62.86 >> -601)) {
            if (true && true) {
            while (true) {
            try {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e96) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        if (true && (null % 14.24)) {
            try {
            return (-423 ^ 251);
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        }
        return null;
    }
    private static long __cfwr_aux873(short __cfwr_p0, Object __cfwr_p1) {
        if (true && false) {
            try {
            return ((29.67f + -863) - -15.58);
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        }
        short __cfwr_obj96 = null;
        return 880L;
    }
    public static long __cfwr_process161(Character __cfwr_p0, double __cfwr_p1) {
        char __cfwr_var23 = 'Z';
        return -820L;
    }
}