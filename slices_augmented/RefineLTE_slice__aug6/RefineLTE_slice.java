/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineLTE_slice {
    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
        while ((('x' ^ 84.46) | null)) {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 2; __cfwr_i34++) {
            for (int __cfwr_i64 = 0; __cfwr_i64 < 3; __cfwr_i64++) {
            if (false && (true ^ 16.23)) {
            for (int __cfwr_i79 = 0; __cfwr_i79 < 8; __cfwr_i79++) {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 6; __cfwr_i55++) {
            Object __cfwr_var88 = null;
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }

    // The reason for the parsing is so that the Value Checker
    // can't figure it out but normal humans can.

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (b <= test) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (b <= a) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }

    @Positive
  void testLTEL(@LTEqLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (b <= test) {
    @Positive
      @LTEqLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (b <= a) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }

    public float __cfwr_helper833() {
        return null;
        Integer __cfwr_data1 = null;
        Integer __cfwr_val6 = null;
        return 49.20f;
    }
}