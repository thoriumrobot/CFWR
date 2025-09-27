/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayAssignmentSameLen_slice {
// Source-based slice around line 19
// Method: ArrayAssignmentSameLen#test1(int[],int[],int)

    i_index = index;
  }

  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
        for (int __cfwr_i42 = 0; __cfwr_i42 < 7; __cfwr_i42++) {
            for (int __cfwr_i97 = 0; __cfwr_i97 < 3; __cfwr_i97++) {
            if ((null / true) || (true - false)) {
            for (int __cfw
        for (int __cfwr_i59 = 0; __cfwr_i59 < 9; __cfwr_i59++) {
            return -37.04;
        }
r_i78 = 0; __cfwr_i78 < 2; __cfwr_i78++) {
            if (true && (-92.05 << 101)) {
            if (((null >> null) + (10L - null)) && true) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 4; __cfwr_i11++) {
            return 1.41;
        }
        }
        }
        }
        }
        }
        }

    int[] array = a;
    @LTLengthOf(
        value = {"array", "b"},
        offset = {"0", "-3"})
    // :: error: (assignment)
    int i = index;
  }

  void test2(int[] a, int[] b, @LTLengthOf("#1") int i) {
    int[] c = a;
    // :: error: (assignment)
    @LTLengthOf(value = {"c", "b"}) int x = i;
    @LTLengthOf("c") int y = i;
  }

  void test3(int[] a, @LTLengthOf("#1") int i, @NonNegative int x) {

    private Boolean __cfwr_func276(Object __cfwr_p0) {
        for (int __cfwr_i32 = 0; __cfwr_i32 < 6; __cfwr_i32++) {
            if (true && false) {
            for (int __cfwr_i11 = 0; __cfwr_i11 < 6; __cfwr_i11++) {
            try {
            try {
            Long __cfwr_elem78 = null;
        } catch (Exception __cfwr_e54) {
            // ignore
        }
        } catch (Exception __cfwr_e30) {
            // ignore
        }
        }
        }
        }
        try {
            if (true || true) {
            try {
            for (int __cfwr_i37 = 0; __cfwr_i37 < 6; __cfwr_i37++) {
            for (int __cfwr_i16 = 0; __cfwr_i16 < 2; __cfwr_i16++) {
            while (true) {
            try {
            while (true) {
            if (true && ((null + -729) % 84L)) {
            while (false) {
            if (true || ((null & null) & 551)) {
            while (true) {
            return ('C' + (false >> 155L));
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        } catch (Exception __cfwr_e68) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        return null;
    }
}