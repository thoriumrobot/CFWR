/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ConstantArrays_slice {
    @Positive
  void basic_test() {
        while (true) {
            try {
            try {
            if (true && true) {
            int __cfwr_data53 = -939;
        }
        } catch (Exception __cfwr_e88) {
            // ignore
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
            break; // Prevent infinite loops
        }

    @Positive
    int[] b = new int[4];
    @Positive
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: 
        if (true && true) {
            try {
            return null;
        } catch (Exception __cfwr_e59) {
            // ignore
        }
        }
(array.initializer)::error: (assignment)
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

    private static Long __cfwr_handle637(char __cfwr_p0, Object __cfwr_p1, boolean __cfwr_p2) {
        try {
            for (int __cfwr_i90 = 0; __cfwr_i90 < 4; __cfwr_i90++) {
            while (true) {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 4; __cfwr_i31++) {
            try {
            byte __cfwr_val2 = null;
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        return null;
    }
}