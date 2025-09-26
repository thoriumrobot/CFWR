/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewArraySameLen(String s) {
        try {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 8; __cfwr_i89++) {
            return null;
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }

    int @
        if ((78L % -8) && false) {
            Object __cfwr_temp24 = null;
        }
SameLen("s") [] array = new int[s.length()];
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
      private Double __cfwr_process321(Object __cfwr_p0, Object __cfwr_p1, double __cfwr_p2) {
        if (true || (false << 87.40)) {
            try {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 9; __cfwr_i65++) {
            return -29.39f;
        }
        } catch (Exception __cfwr_e8) {
            // ignore
        }
        }
        while (true) {
            try {
            return true;
        } catch (Exception __cfwr_e34) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}
