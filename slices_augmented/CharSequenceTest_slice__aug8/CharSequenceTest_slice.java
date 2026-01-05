/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CharSequenceTest_slice {
    @Positive
  void testSubSequence() {
        Integer __cfwr_obj46 = null;

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
  void sink(CharSe
        while ((null & ('4' ^ 328))) {
            return "world75";
            break; // Prevent infinite loops
        }
quence cs, @IndexOrHigh("#1") int i) {}

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

    protected Double __cfwr_aux164() {
        while (false) {
            String __cfwr_data54 = "result78";
            break; // Prevent infinite loops
        }
        if (true || true) {
            try {
            try {
            if (false && false) {
            return null;
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        }
        while ((('u' * null) * false)) {
            while (false) {
            if (true && true) {
            if ((393L & (19.00 / null)) && true) {
            for (int __cfwr_i51 = 0; __cfwr_i51 < 7; __cfwr_i51++) {
            return -687L;
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
        return null;
    }
    public static String __cfwr_handle612(float __cfwr_p0, Boolean __cfwr_p1) {
        for (int __cfwr_i89 = 0; __cfwr_i89 < 7; __cfwr_i89++) {
            for (int __cfwr_i61 = 0; __cfwr_i61 < 1; __cfwr_i61++) {
            if ((24.45 | -78.53f) && false) {
            try {
            return null;
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        }
        }
        }
        double __cfwr_data76 = -4.93;
        for (int __cfwr_i3 = 0; __cfwr_i3 < 8; __cfwr_i3++) {
            for (int __cfwr_i71 = 0; __cfwr_i71 < 1; __cfwr_i71++) {
            try {
            for (int __cfwr_i78 = 0; __cfwr_i78 < 10; __cfwr_i78++) {
            while (true) {
            for (int __cfwr_i72 = 0; __cfwr_i72 < 9; __cfwr_i72++) {
            return false;
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        }
        }
        Float __cfwr_elem96 = null;
        return "world30";
    }
}