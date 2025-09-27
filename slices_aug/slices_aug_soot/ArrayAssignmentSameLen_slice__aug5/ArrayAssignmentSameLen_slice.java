/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayAssignmentSameLen_slice {
// Source-based slice around line 19
// Method: ArrayAssignmentSameLen#test1(int[],int[],int)

    i_index = index;
  }

  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
        short __cfwr_data97 = null;

    int[] array = a;
    @LTLengthOf(
        value = {"array", "b"},
        offset = {"0", "-3"})
    // :: error: (assignment)
    int i = index;
  }

  void test2(int[] a
        while (false) {
            if (true && false) {
            return -20.38f;
        }
            break; // Prevent infinite loops
        }
, int[] b, @LTLengthOf("#1") int i) {
    int[] c = a;
    // :: error: (assignment)
    @LTLengthOf(value = {"c", "b"}) int x = i;
    @LTLengthOf("c") int y = i;
  }

  void test3(int[] a, @LTLengthOf("#1") int i, @NonNegative int x) {

    static Integer __cfwr_aux862() {
        if (((null * -60.29) | 90.17f) && (null - (null % 31.39f))) {
            if (true || false) {
            return null;
        }
        }
        return null;
    }
}