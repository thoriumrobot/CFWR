/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CharSequenceTest_slice {
    @Positive
  void testSubSequence() {
        return "test79";

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

    double __cfwr_process206(short __cfwr_p0, String __cfwr_p1, double __cfwr_p2) {
        if (false && (-188L ^ -148L)) {
            short __cfwr_data12 = null;
        }
        for (int __cfwr_i14 = 0; __cfwr_i14 < 1; __cfwr_i14++) {
            return null;
        }
        if (true || (-1.03 | -4.40)) {
            try {
            float __cfwr_val71 = (('O' + 75.51) | null);
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        }
        int __cfwr_obj65 = (null & (81.54f - 83.50));
        return 61.06;
    }
    public static Character __cfwr_handle866() {
        Boolean __cfwr_data36 = null;
        return ((757 >> '5') / -2);
        int __cfwr_data46 = -344;
        try {
            Integer __cfwr_entry61 = null;
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        return null;
    }
}