/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class IntroAdd_slice {
    @Positive
  void test(int[] arr) {
        while (true) {
            try {
            Object __cfwr_elem35 = null;
        } catch (Exception __cfwr_e20) {
            // ignore
        }
            break; // Prevent infinite loops
        }

    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int a = 3;
    @Positive
  }

    @Positive
  void test(int[] arr, @LTLengthOf({"#1"}) int a) {
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int c = a + 1;
    @Positive
    @LTEqLengthOf({"arr"}) int c1 = a + 1;
    @Positive
    @LTLengthOf({"arr"}) int d = a + 0;
    @Positive
    @LTLengthOf({"arr"}) int e = a + (-7);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"arr"}) int f = a + 7;
    @Positive
  }

    protected static float __cfwr_temp861(Integer __cfwr_p0) {
        while (false) {
            while (true) {
            for (int __cfwr_i4 = 0; __cfwr_i4 < 6; __cfwr_i4++) {
            try {
            Long __cfwr_temp90 = null;
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return 78.26f;
    }
}