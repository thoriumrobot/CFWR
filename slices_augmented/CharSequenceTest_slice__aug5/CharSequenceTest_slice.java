/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CharSequenceTest_slice {
    @Positive
  void testSubSequence() {
        for (int __cfwr_i88 = 0; __cfwr_i88 < 3; __cfwr_i88++) {
            if (false && (null & null)) {
            while ((-2.60 % ('A' - null))) {
            while (false) {
            for (int __cfwr_i45 = 0; __cfwr_i45 < 4; __cfwr_i45++) {
            while (true) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 4; __cfwr_i11++) {
            try {
            int __cfwr_data43 = 807
        while (false) {
            Integer __cfwr_temp43 = null;
            break; // Prevent infinite loops
        }
;
        } catch (Exception __cfwr_e15) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
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

    protected int __cfwr_calc777(float __cfwr_p0, short __cfwr_p1) {
        while (false) {
            if ((-75 & (null | -444L)) || ((62.03f / null) % (53.26f >> null))) {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 7; __cfwr_i34++) {
            Long __cfwr_temp81 = null;
        }
        }
            break; // Prevent infinite loops
        }
        while (true) {
            while (true) {
            for (int __cfwr_i32 = 0; __cfwr_i32 < 9; __cfwr_i32++) {
            Long __cfwr_item69 = null;
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return -786;
    }
    protected Object __cfwr_calc83(Boolean __cfwr_p0) {
        Integer __cfwr_val81 = null;
        while (true) {
            while (true) {
            if ((null - (-65 * null)) && ((-127L & -50.90) * (-760L % 38.91f))) {
            while ((null % null)) {
            while ((null & null)) {
            for (int __cfwr_i38 = 0; __cfwr_i38 < 7; __cfwr_i38++) {
            if (('X' << 951L) && true) {
            while (false) {
            while (true) {
            Long __cfwr_node59 = null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        double __cfwr_obj26 = 27.70;
        return null;
    }
}