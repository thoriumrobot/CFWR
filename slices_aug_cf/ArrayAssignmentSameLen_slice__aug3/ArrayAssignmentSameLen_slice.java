/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        while ((('j' * null) + 46.10)) {
            try {
            while (false) {
            try {
     
        Float __cfwr_elem33 = null;
       return (-77 << null);
        } catch (Exception __cfwr_e42) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }
            break; // Prevent infinite loops
        }

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
      protected static byte __cfwr_process198(int __cfwr_p0) {
        while (false) {
            if (false || (288 % 50.70f)) {
            for (int __cfwr_i39 = 0; __cfwr_i39 < 8; __cfwr_i39++) {
            if (false || (-760L + null)) {
            if (true && ((362 / true) << true)) {
            return null;
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
        long __cfwr_obj15 = -924L;
        return ((null | true) / 54.86);
    }
}
