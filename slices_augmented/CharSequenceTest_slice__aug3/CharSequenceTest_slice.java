/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class CharSequenceTest_slice {
    @Positive
  void testSubSequence() {
        while (false) {
            try {
            if (((-490L - null) >> 900L) || true) {
            try {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 1; __cfwr_i59++) {
            while (true) {
            Long __cfwr_temp46 = null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        }
    
        return null;
    } catch (Exception __cfwr_e89) {
            // ignore
        }
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

    public boolean __cfwr_util75() {
        boolean __cfwr_node67 = ('o' - null);
        for (int __cfwr_i42 = 0; __cfwr_i42 < 5; __cfwr_i42++) {
            try {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 10; __cfwr_i3++) {
            if ((false - 312L) && true) {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 1; __cfwr_i9++) {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 3; __cfwr_i9++) {
            if (false || ((196 - -765L) % -93.59)) {
            float __cfwr_entry56 = -15.25f;
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        }
        for (int __cfwr_i34 = 0; __cfwr_i34 < 7; __cfwr_i34++) {
            return null;
        }
        while (false) {
            char __cfwr_elem57 = 'D';
            break; // Prevent infinite loops
        }
        return (31 - null);
    }
    byte __cfwr_calc795() {
        try {
            try {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 3; __cfwr_i52++) {
            if ((-36.15f | (360 * null)) || false) {
            try {
            for (int __cfwr_i52 = 0; __cfwr_i52 < 8; __cfwr_i52++) {
            try {
            while (true) {
            if ((-94.75 << -72.65) && (974 | 836)) {
            try {
            if (false && ((50.15 ^ 300) % true)) {
            try {
            try {
            return 'B';
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        } catch (Exception __cfwr_e6) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e77) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e37) {
            // ignore
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        while (true) {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        try {
            for (int __cfwr_i13 = 0; __cfwr_i13 < 8; __cfwr_i13++) {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 6; __cfwr_i43++) {
            if (false || true) {
            for (int __cfwr_i30 = 0; __cfwr_i30 < 4; __cfwr_i30++) {
            while ((null & (false | null))) {
            while (true) {
            return "test94";
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        while (true) {
            return false;
            break; // Prevent infinite loops
        }
        return (('5' / false) + -69.66);
    }
}