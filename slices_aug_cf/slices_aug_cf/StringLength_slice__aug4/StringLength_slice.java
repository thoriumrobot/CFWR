/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewArraySameLen(String s) {
        for (int __cfwr_i66 = 0; __cfwr_i66 < 1; __cfwr_i66++) {
            while ((59.47f >> (-919 | -374L))) {
            return null;
            break; // Prevent infinite loops
        }
       
        return 'L';
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
      protected float __cfwr_compute369(Double __cfwr_p0) {
        for (int __cfwr_i21 = 0; __cfwr_i21 < 3; __cfwr_i21++) {
            Boolean __cfwr_entry65 = null;
        }
        return -1.90f;
    }
    public static Boolean __cfwr_temp151() {
        Long __cfwr_obj33 = null;
        boolean __cfwr_var30 = (null >> 61);
        return null;
        for (int __cfwr_i82 = 0; __cfwr_i82 < 2; __cfwr_i82++) {
            if ((-58.32f + (163L | 'H')) && (55.13 * -18.07f)) {
            for (int __cfwr_i26 = 0; __cfwr_i26 < 10; __cfwr_i26++) {
            return (99.93 >> -2.80f);
        }
        }
        }
        return null;
    }
}
