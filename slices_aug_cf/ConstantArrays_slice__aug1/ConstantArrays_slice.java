/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ConstantArrays_slice {
  void basic_test() {
        for (int __cfwr_i43 = 0; __cfwr_i43 < 8; __cfwr_i43++) {
            try {
            try {
            while (true) {
            try {
            try {
            if (false || ((-15.06f << -51.59f) | null)) {
            if ((-723 | (null ^ 593)) || false) {
            while (false) {
            if (true && (-19.87 << false)) {
            if (true && true) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e95) {
            // ignore
        }
        } catch (Exception __cfwr_e84) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        } catch (Exception __cfwr_e57) {
            // ignore
        }
        }

    int[] b = new int[4];
    @LTLengthOf("b") int[] a = {0, 1, 2, 3};

    // :: error: (array.initializer)::error: (assignment)
    @LTLengthOf("b") int[] a1 = {0, 1, 2, 4};

    @LTEqLengthOf("b") int[] c = {-1, 4, 3, 1};

    // :: error: (array.initializer)::error: (assignment)
    @LTEqLengthOf("b") int[] c2 = {-1, 4, 5, 1};
  }

  void offset_test() {
    int[] b = new int[4];
    int[] b2 = new int[10];
    @LTLengthOf(
        value = {"b", "b2"},
        offset = {"-2", "5"})
    int[] a = {2, 3, 0};

    @LTLengthOf(
        value = {"b", "b2"},
        offset = {"-2", "5"})
    // :: error: (array.initializer)::error: (assignment)
    int[] a2 = {2, 3, 5};

    // Non-constant offsets don't work correctly. See kelloggm#120.
  }

    Object __cfwr_temp644(char __cfwr_p0, Integer __cfwr_p1) {
        try {
            try {
            return null;
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        } catch (Exception __cfwr_e43) {
            // ignore
        }
        Double __cfwr_temp86 = null;
        return null;
    }
    protected long __cfwr_handle545() {
        return null;
        try {
            try {
            try {
            Boolean __cfwr_result17 = null;
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        } catch (Exception __cfwr_e55) {
            // ignore
        }
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        try {
            try {
            return null;
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        try {
            try {
            try {
            if (false && true) {
            if (true || false) {
            for (int __cfwr_i23 = 0; __cfwr_i23 < 1; __cfwr_i23++) {
            while ((133L / (null * 25.74f))) {
            return null;
            break; // Prevent infinite loops
        }
        }
        }
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        } catch (Exception __cfwr_e63) {
            // ignore
        }
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        return -195L;
    }
}