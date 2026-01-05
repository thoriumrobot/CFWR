/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CharSequenceTest_slice {
    @Positive
  void testSubSequence() {
        while (false) {
            try {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 10; __cfwr_i59++) {
            try {
            Double __cfwr_item48 = null;
        } catch (Exception __cfwr_e10) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e38) {
            // ignore
        }
            break; // Prevent infinite loops
        }

    // Local va
        for (int __cfwr_i77 = 0; __cfwr_i77 < 4; __cfwr_i77++) {
            return 42;
        }
riable used because of https://github.com/kelloggm/checker-framework/issues/165
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

    Long __cfwr_calc127(Boolean __cfwr_p0) {
        try {
            try {
            Character __cfwr_var6 = null;
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        } catch (Exception __cfwr_e41) {
            // ignore
        }
        for (int __cfwr_i34 = 0; __cfwr_i34 < 3; __cfwr_i34++) {
            float __cfwr_data61 = -59.02f;
        }
        try {
            try {
            if ((178L | -585L) && true) {
            for (int __cfwr_i22 = 0; __cfwr_i22 < 4; __cfwr_i22++) {
            Integer __cfwr_obj5 = null;
        }
        }
        } catch (Exception __cfwr_e1) {
            // ignore
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        return null;
    }
    public static short __cfwr_util828(long __cfwr_p0) {
        for (int __cfwr_i68 = 0; __cfwr_i68 < 8; __cfwr_i68++) {
            if (false || (null >> (-394L & null))) {
            for (int __cfwr_i51 = 0; __cfwr_i51 < 3; __cfwr_i51++) {
            try {
            if (true || true) {
            try {
            if (true && false) {
            return null;
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        }
        }
        }
        return null;
    }
}