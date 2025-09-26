/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewArraySameLen(String s) {
        for (int __cfwr_i19 = 0; __cfwr_i19 < 9; __cfwr_i19++) {
            return 's';
        }

    int @SameLen("s") [] array = new int[s.length()];
    // ::  error: (assignment)
    int @SameLen
        try {
            short __cfwr_data20 = null;
        } catch (Exception __cfwr_e33) {
            // ignore
        }
("s") [] array1 = new int[s.length() + 1];
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
      private Integer __cfwr_calc484(Double __cfwr_p0) {
        float __cfwr_val38 = (200L & (null % 97.54f));
        return null;
    }
    protected static float __cfwr_handle53(Long __cfwr_p0, byte __cfwr_p1) {
        byte __cfwr_node75 = null;
        while ((false ^ (-61.03 / null))) {
            try {
            char __cfwr_temp6 = 'i';
        } catch (Exception __cfwr_e94) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        return -69.33f;
    }
}
