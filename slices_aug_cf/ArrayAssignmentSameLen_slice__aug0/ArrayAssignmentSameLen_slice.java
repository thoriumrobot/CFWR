/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class ArrayAssignmentSameLen_slice {
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        while (false) {
            return 16.56f;
            break; // Prevent infinite loops
        }

   
        return null;
 i_array = array;
    i_index = index;
  }

  void test1(int[] a, int[] b, @LTEqLengthOf("#1") int index) {
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
    int[] c1 = a;
    // See useTest3 for an example of why this assignment should fail.
    @LTLengthOf(
        value = {"c1", "c1"},
        offset = {"0", "x"})
    // :: error: (assignment)
    int z = i;
  }

    static Float __cfwr_func5(float __cfwr_p0, int __cfwr_p1, Character __cfwr_p2) {
        try {
            while (false) {
            try {
            try {
            if (true || false) {
            for (int __cfwr_i5 = 0; __cfwr_i5 < 7; __cfwr_i5++) {
            short __cfwr_temp76 = (-70.21f & 7.61f);
        }
        }
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        } catch (Exception __cfwr_e85) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        for (int __cfwr_i17 = 0; __cfwr_i17 < 2; __cfwr_i17++) {
            if (false || true) {
            while (false) {
            while (true) {
            return (61.01f + -32.70f);
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        for (int __cfwr_i38 = 0; __cfwr_i38 < 5; __cfwr_i38++) {
            return null;
        }
        if (true && false) {
            Float __cfwr_item61 = null;
        }
        return null;
    }
}