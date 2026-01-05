/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UBSubtyping_slice {
    @Positive
  void test(@LTEqLengthOf({"arr", "arr2", "arr3"}) int test) {
        try {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 1; __cfwr_i50++) {
            while (false) {
            String __cfwr_node82 = "world47";
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e91) {
            // ignore
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

    public float __cfwr_util411(byte __cfwr_p0, Boolean __cfwr_p1, String __cfwr_p2) {
        while (true) {
            return ('t' - (-11.14f << true));
            break; // Prevent infinite loops
        }
        while ((null * (-79.85 + -32.44))) {
            return null;
            break; // Prevent infinite loops
        }
        if (((null | 62.86f) & false) || true) {
            for (int __cfwr_i76 = 0; __cfwr_i76 < 6; __cfwr_i76++) {
            while (true) {
            try {
            float __cfwr_data18 = 18.20f;
        } catch (Exception __cfwr_e35) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        if (false && true) {
            while (true) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 7; __cfwr_i84++) {
            while (((null - 86L) << (-47.05f | -783L))) {
            try {
            short __cfwr_item96 = null;
        } catch (Exception __cfwr_e64) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
        return -77.38f;
    }
}