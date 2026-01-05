/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineLT_slice {
    @Positive
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        while (false) {
            Float __cfwr_obj50 = null;
            break; // Prevent infinite loops
        }

    @Positive
    int b = 2;
    @Positive
    if (b < test) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (b < a3) {
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
  void testLTEL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    @Positive
    int b = 2;
    @Positive
    if (b < test) {
    @Positive
      @LTEqLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf("arr") int c1 = b;

    @Positive
    if (b < a) {
    @Positive
      int potato = 7;
    @Positive
    } else {
      // :: error: (assignment)
    @Positive
      @LTEqLengthOf("arr") int d = b;
    @Positive
    }
    @Positive
  }

    public Float __cfwr_compute647(Float __cfwr_p0) {
        for (int __cfwr_i93 = 0; __cfwr_i93 < 6; __cfwr_i93++) {
            if (false && true) {
            if (false && false) {
            Float __cfwr_node95 = null;
        }
        }
        }
        String __cfwr_obj36 = "test76";
        return null;
    }
    byte __cfwr_temp847() {
        short __cfwr_obj75 = null;
        return null;
    }
    private static String __cfwr_helper49(Character __cfwr_p0) {
        for (int __cfwr_i71 = 0; __cfwr_i71 < 5; __cfwr_i71++) {
            return (506 % (null & null));
        }
        try {
            Integer __cfwr_entry8 = null;
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        return (null / 864L);
        byte __cfwr_node54 = null;
        return "data49";
    }
}