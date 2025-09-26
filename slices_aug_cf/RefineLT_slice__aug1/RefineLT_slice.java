/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RefineLT_slice {
  void testLTL(@LTLengthOf("arr") int test, @LTLengthOf("arr") int a, @LTLengthOf("arr") int a3) {
        for (int __cfwr_i11 = 0; _
        try {
            if (true && false) {
            int __cfwr_temp56 = -954;
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
_cfwr_i11 < 9; __cfwr_i11++) {
            Boolean __cfwr_result98 = null;
        }

    int b = 2;
    if (b < test) {
      @LTLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTLengthOf("arr") int c1 = b;

    if (b < a3) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTLengthOf("arr") int d = b;
    }
  }

  void testLTEL(@LTLengthOf("arr") int test) {
    // :: error: (assignment)
    @LTEqLengthOf("arr") int a = Integer.parseInt("1");

    // :: error: (assignment)
    @LTEqLengthOf("arr") int a3 = Integer.parseInt("3");

    int b = 2;
    if (b < test) {
      @LTEqLengthOf("arr") int c = b;
    }
    // :: error: (assignment)
    @LTEqLengthOf("arr") int c1 = b;

    if (b < a) {
      int potato = 7;
    } else {
      // :: error: (assignment)
      @LTEqLengthOf("arr") int d = b;
    }
  }

    protected static String __cfwr_util40(Integer __cfwr_p0, byte __cfwr_p1, long __cfwr_p2) {
        for (int __cfwr_i21 = 0; __cfwr_i21 < 1; __cfwr_i21++) {
            while (((false - -79.07) * 54.97)) {
            double __cfwr_temp83 = -68.49;
            break; // Prevent infinite loops
        }
        }
        return "world13";
    }
}