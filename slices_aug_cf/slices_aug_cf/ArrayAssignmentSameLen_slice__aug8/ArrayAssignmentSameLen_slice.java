/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  ArrayAssignmentSameLen(int[] array, @IndexFor("#1") int index) {
        for (int __cfwr_i20 = 0; __cfwr_i20 < 4; __cfwr_i20++) {
            long __cfwr_item27 = -53L;
      
        try {
            return -57.99f;
        } catch (Exception __cfwr_e14) {
            // ignore
        }
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
      private static Long __cfwr_compute693(String __cfwr_p0) {
        Float __cfwr_data78 = null;
        try {
            try {
            if (false || false) {
            for (int __cfwr_i2 = 0; __cfwr_i2 < 7; __cfwr_i2++) {
            for (int __cfwr_i83 = 0; __cfwr_i83 < 7; __cfwr_i83++) {
            for (int __cfwr_i96 = 0; __cfwr_i96 < 9; __cfwr_i96++) {
            while (false) {
            try {
            if (('f' & -42.73) || false) {
            return -956L;
        }
        } catch (Exception __cfwr_e20) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        } catch (Exception __cfwr_e5) {
            // ignore
        }
        } catch (Exception __cfwr_e39) {
            // ignore
        }
        for (int __cfwr_i77 = 0; __cfwr_i77 < 7; __cfwr_i77++) {
            Integer __cfwr_data60 = null;
        }
        return null;
    }
    static String __cfwr_aux332() {
        if (false || ('d' - null)) {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 6; __cfwr_i41++) {
            return null;
        }
        }
        try {
            Float __cfwr_result66 = null;
        } catch (Exception __cfwr_e74) {
            // ignore
        }
        return null;
        for (int __cfwr_i36 = 0; __cfwr_i36 < 7; __cfwr_i36++) {
            if (false || true) {
            if (false && true) {
            for (int __cfwr_i25 = 0; __cfwr_i25 < 7; __cfwr_i25++) {
            if (true || false) {
            try {
            while (false) {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 6; __cfwr_i55++) {
            try {
            while (true) {
            Integer __cfwr_data30 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e57) {
            // ignore
        }
        }
        }
        }
        }
        }
        return "value82";
    }
    float __cfwr_func425(byte __cfwr_p0) {
        return "hello78";
        while ((('T' | null) * (null + 534L))) {
            for (int __cfwr_i60 = 0; __cfwr_i60 < 6; __cfwr_i60++) {
            if (true || true) {
            try {
            if (((null & false) / -518L) && false) {
            return null;
        }
        } catch (Exception __cfwr_e21) {
            // ignore
        }
        }
        }
            break; // Prevent infinite loops
        }
        return 5.95f;
    }
}
