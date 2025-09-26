/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UBSubtyping_slice {
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        try {
            Integer __cfwr_obj78 = null;
        } catch (Exception __cfwr_e19) {
            // ignore
        }

    // :: error: (assignment)
    @LTEqLengthOf({"arr"}) int a = 1;
    // :: error: (assignment)
    @LTLengthOf({"arr"}) int a1 = 1;

    // :: error: (assignment)
    @LTLengthOf({"arr"}) int b = a;
    @UpperBoundUnknown int d = a;

    // :: error: (assignment)
    @LTLengthOf({"arr2"}) int g = a;

    // :: error: (assignment)
    @LTEqLengthOf({"arr", "arr2", "arr3"}) int h = 2;

    @LTEqLengthOf({"arr", "arr2"}) int h2 = test;
    @LTEqLengthOf({"arr"}) int i = test;
    @LTEqLengthOf({"arr", "arr3"}) int j = test;
  }

    private long __cfwr_util823(byte __cfwr_p0) {
        try {
            return null;
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        try {
            for (int __cfwr_i21 = 0; __cfwr_i21 < 9; __cfwr_i21++) {
            if ((null | -517L) || true) {
            for (int __cfwr_i94 = 0; __cfwr_i94 < 9; __cfwr_i94++) {
            while (true) {
            if ((298L >> null) && true) {
            Float __cfwr_result85 = null;
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        return -214L;
    }
    public char __cfwr_handle244() {
        int __cfwr_item29 = 210;
        return 'b';
    }
}