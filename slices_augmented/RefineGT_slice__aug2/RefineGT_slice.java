/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineGT_slice {
    @Positive
  void testLTL(@LTLengthOf("arr") int test) {
        Integer __cfwr_temp39 = null;

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
    if (test > b) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (a > b) {
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
    if (test > b) {
    @Positive
      @LTLengthOf("arr") int c = b;
    @Positive
    }
    // :: error: (assignment)
    @Positive
    @LTLengthOf("arr") int c1 = b;

    @Positive
    if (a > b) {
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

    Character __cfwr_calc728(Character __cfwr_p0, char __cfwr_p1, double __cfwr_p2) {
        if (((null ^ 112L) % false) && false) {
            try {
            if (true && (null + false)) {
            short __cfwr_obj95 = null;
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        }
        return null;
    }
    static boolean __cfwr_aux834(long __cfwr_p0) {
        try {
            for (int __cfwr_i77 = 0; __cfwr_i77 < 9; __cfwr_i77++) {
            if (false || true) {
            if (false && true) {
            return (15.49f & (null | null));
        }
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        Double __cfwr_node96 = null;
        for (int __cfwr_i59 = 0; __cfwr_i59 < 8; __cfwr_i59++) {
            for (int __cfwr_i75 = 0; __cfwr_i75 < 1; __cfwr_i75++) {
            return -8.31;
        }
        }
        return false;
    }
    private byte __cfwr_calc845(Character __cfwr_p0) {
        for (int __cfwr_i37 = 0; __cfwr_i37 < 6; __cfwr_i37++) {
            try {
            try {
            if (false || true) {
            float __cfwr_elem3 = ((-39.63f | 75.32) * (true & 281L));
        }
        } catch (Exception __cfwr_e40) {
            // ignore
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        }
        return null;
        float __cfwr_val28 = -73.46f;
        return (673 ^ (-23.39 | null));
    }
}