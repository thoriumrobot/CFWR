/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CharSequenceTest_slice {
    @Positive
  void testSubSequence() {
        if ((false + false) || false) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 7; __cfwr_i84++) {
            while (true) {
            for (int __cfwr_i47 = 0; __cfwr_i47 < 5; __cfwr_i47++) {
            try {
            while ((-3.48 << false)) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
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

    private static String __cfwr_calc740(Integer __cfwr_p0, Float __cfwr_p1) {
        try {
            try {
            while (false) {
            for (int __cfwr_i2 = 0; __cfwr_i2 < 4; __cfwr_i2++) {
            return 968;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        if (false && false) {
            try {
            try {
            try {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 3; __cfwr_i32++) {
            return 134L;
        }
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        } catch (Exception __cfwr_e58) {
            // ignore
        }
        }
        try {
            Object __cfwr_val41 = null;
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        while (((-756L * -854) % -170L)) {
            return null;
            break; // Prevent infinite loops
        }
        return "value94";
    }
}