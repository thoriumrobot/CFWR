/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UBSubtyping_slice {
    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        for (int __cfwr_i10 = 0; __cfwr_i10 < 8; __cfwr_i10++) {
            return true;
        }

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int b = a;
    @Positive
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @Positive
    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @Positive
    @LTEqLengthOf({"arr"}) int i = test;
    @Positive
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
    @Positive
  }

    private boolean __cfwr_proc746() {
        for (int __cfwr_i24 = 0; __cfwr_i24 < 10; __cfwr_i24++) {
            try {
            try {
            return (-166 + (null << true));
        } catch (Exception __cfwr_e66) {
            // ignore
        }
        } catch (Exception __cfwr_e51) {
            // ignore
        }
        }
        for (int __cfwr_i49 = 0; __cfwr_i49 < 2; __cfwr_i49++) {
            for (int __cfwr_i3 = 0; __cfwr_i3 < 4; __cfwr_i3++) {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 5; __cfwr_i35++) {
            Double __cfwr_val3 = null;
        }
        }
        }
        if ((null << false) && (null + 17.60f)) {
            while (true) {
            return "item7";
            break; // Prevent infinite loops
        }
        }
        return true;
    }
    protected Character __cfwr_aux8(float __cfwr_p0, float __cfwr_p1, long __cfwr_p2) {
        for (int __cfwr_i94 = 0; __cfwr_i94 < 3; __cfwr_i94++) {
            return null;
        }
        return null;
        return null;
    }
}