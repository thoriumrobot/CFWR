/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        try {
            return null;
        } catch (Exception __cfwr_e84) {
            // ignore
        
        return -490;
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
      static Double __cfwr_temp482(double __cfwr_p0, String __cfwr_p1) {
        while (false) {
            return null;
            break; // Prevent infinite loops
        }
        while (false) {
            while (false) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 2; __cfwr_i14++) {
            try {
            if (true && false) {
            if (true || true) {
            if (true && false) {
            Object __cfwr_item31 = null;
        }
        }
        }
        } catch (Exception __cfwr_e36) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    public static Integer __cfwr_temp887(Character __cfwr_p0, String __cfwr_p1) {
        while ((null / -919L)) {
            for (int __cfwr_i24 = 0; __cfwr_i24 < 2; __cfwr_i24++) {
            if ((('G' / -55.17f) >> (-454L + -517)) || true) {
            return "hello43";
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
