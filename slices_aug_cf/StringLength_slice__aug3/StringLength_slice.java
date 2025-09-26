/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewArraySameLen(String s) {
        try {
            for (int __cfwr_i28 = 0; __cfwr_i28 < 7; __cfwr_i28++) {
            try {
            return -46.73f;
        } catch (Exception __cfwr_e98) {
            // ignore
        }
        for (int __cfwr_i19 = 0; __cfwr_i19 < 3; __cfwr_i19++) {
            for (int __cfwr_i9 = 0; __cfwr_i9 < 10; __cfwr_i9++) {
            while (false) {
            return "hello92";
            break; // Prevent infinite loops
        }
        }
        }

        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }

    int @SameLen("s") [] array = new int[s.length()];
    // ::  error: (assignment)
    int @SameLen("s") [] array1 = new int[s.length() + 1];
  }

  void testStringAssignSameLen(String s, String r) {
    @SameLen("s") String t = s;
    // ::  error: (assignment)
    @SameLen("s") String tN = r;
  }

  void testStringLenEqualSameLen(String s, String r) {
    if (s.length() == r.length()) {
      @SameLen("s") String tN = r;
    }
  }

  void testStringEqualSameLen(String s, String r) {
    if (s == r) {
      @SameLen("s") String tN = r;
    }
  }

  void testOffsetRemoval(
      String s,
      String t,
      @LTLengthOf(value = "#1", offset = "#2.length()") int i,
      @LTLengthOf(value = "#2") int j,
      int k) {
    @LTLengthOf("s") int ij = i + j;
    // ::  error: (assignment)
    @LTLengthOf("s") int ik = i + k;
  }

  void testLengthDivide(@MinLen(1) String s) {
    @IndexFor("s") int i = s.length() / 2;
  }

  void testAddDivide(@MinLen(1) String s, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @IndexFor("s") int ij = (i + j) / 2;
  }

  void testRandomMultiply(@MinLen(1) String s, Random r) {
    @LTLengthOf("s") int i = (int) (Math.random() * s.length());
    @LTLengthOf("s") int j = (int) (r.nextDouble() * s.length());
  }

  void testNotEqualLength(String s, @IndexOrHigh("#1") int i, @IndexOrHigh("#1") int j) {
    if (i != s.length()) {
      @IndexFor("s") int in = i;
      // ::  error: (assignment)
      @IndexFor("s") int jn = j;
    }
  }

  void testLength(String s) {
    @IndexOrHigh("s") int i = s.length();
    @LTLengthOf("s") int j = s.length() - 1;
      static double __cfwr_calc858() {
        if (true || (27.91 % 134)) {
            return null;
        }
        while (false) {
            try {
            if (true && true) {
            for (int __cfwr_i48 = 0; __cfwr_i48 < 9; __cfwr_i48++) {
            try {
            return null;
        } catch (Exception __cfwr_e28) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e95) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        while (((null % '7') * 3.44f)) {
            while (true) {
            if (('W' - -43.63) || true) {
            if (false && (('C' << true) * null)) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        try {
            return null;
        } catch (Exception __cfwr_e60) {
            // ignore
        }
        return 65.74;
    }
    protected static Float __cfwr_handle486(Character __cfwr_p0, float __cfwr_p1) {
        try {
            return null;
        } catch (Exception __cfwr_e97) {
            // ignore
        }
        return 22.80;
        return null;
    }
}
