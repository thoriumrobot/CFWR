/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ConstantArrays_slice {
    @Positive
  void basic_test() {
        try {
            for (int __cfwr_i19 = 0; __cfwr_i19 < 10; __cfwr_i19++) {
            int __cfwr_node96 = (78.97f % (null & null));
        }
        } catch (Exception __cfwr_e98) {
            // ignore
        }

    @Positive
    int[] b = new int[4];
    @Positive
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
    @Positive
    @LTLengthOf("b") int[] a1 = {0, 1, 2, 4};

    @Positive
        return 538;

    @LTEqLengthOf("b") int[] c = {-1, 4, 3, 1};

    // :: error: (array.initializer)::error: (assignment)
    @Positive
    @LTEqLengthOf("b") int[] c2 = {-1, 4, 5, 1};
    @Positive
  }

    @Positive
  void offset_test() {
    @Positive
    int[] b = new int[4];
    @Positive
    int[] b2 = new int[10];
    @Positive
        value = {"b", "b2"},
    @Positive
        offset = {"-2", "5"})
    @Positive
    int[] a = {2, 3, 0};

    @Positive
        value = {"b", "b2"},
    @Positive
        offset = {"-2", "5"})
    // :: error: (array.initializer)::error: (assignment)
    @Positive
    int[] a2 = {2, 3, 5};

    // Non-constant offsets don't work correctly. See kelloggm#120.
    @Positive
  }

    protected Character __cfwr_calc512(String __cfwr_p0, double __cfwr_p1, String __cfwr_p2) {
        Character __cfwr_entry63 = null;
        try {
            try {
            return -11.52f;
        } catch (Exception __cfwr_e49) {
            // ignore
        }
        } catch (Exception __cfwr_e63) {
            // ignore
        }
        return null;
    }
    private static double __cfwr_func859(String __cfwr_p0, Boolean __cfwr_p1) {
        if (false || true) {
            return null;
        }
        return 86.05;
    }
}