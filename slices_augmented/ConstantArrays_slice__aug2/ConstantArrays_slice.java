/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ConstantArrays_slice {
    @Positive
  void basic_test() {
        if (false || false) {
            return null;
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

    static Object __cfwr_compute22(Character __cfwr_p0, int __cfwr_p1) {
        try {
            try {
            while ((null << 45.91)) {
            if (false && (('9' + false) / (205L + null))) {
            return (261L & 'O');
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e90) {
            // ignore
        }
        } catch (Exception __cfwr_e56) {
            // ignore
        }
        byte __cfwr_var84 = null;
        return null;
    }
}