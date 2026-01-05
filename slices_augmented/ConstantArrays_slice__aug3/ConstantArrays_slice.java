/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ConstantArrays_slice {
    @Positive
  void basic_test() {
        if (true && true) {
            Boolean __cfwr_val93 = null;
        }

    @Positive
    int[] b = new int[4];
    @Positive
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
    @Positive
    @LTLengthOf("b") int[] a1 = {0, 1, 2, 4};

    @Positive
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

    private byte __cfwr_temp857() {
        if (false && true) {
            try {
            try {
            while (('u' & (61.47 << null))) {
            return 'x';
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e12) {
            // ignore
        }
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        }
        return null;
    }
    static Long __cfwr_proc553() {
        Object __cfwr_entry61 = null;
        try {
            Integer __cfwr_obj72 = null;
        } catch (Exception __cfwr_e22) {
            // ignore
        }
        return null;
    }
    public static boolean __cfwr_process523(Character __cfwr_p0) {
        return (('g' - null) * (-974 + -1.47));
        return true;
    }
}