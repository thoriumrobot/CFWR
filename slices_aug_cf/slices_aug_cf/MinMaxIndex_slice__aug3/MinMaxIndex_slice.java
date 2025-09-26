/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void indexForOrHigh(String str, @IndexFor("#1") int i1, @IndexOrHigh("#1") int i2) {
        boolean __cfwr_val80 = false;

    str.substring(Math.max(i1, i2));
    str.substring(Math.min(i1, i2));
    // :: error: (argument)
    str.charAt(Math.max(i1, i2));
    str.charAt(Math.min(i1, i2));
  }

  // max does not work with different sequences, min does
  void twoSequences(String str1, String str2, @IndexFor("#1") int i1, @IndexFor("#2") int i2) {
    // :: error: (argument)
    str1.charAt(Math.max(i1, i2));
    str1.charAt(Math.min(i1, i2));
      public Float __cfwr_util200(double __cfwr_p0, Integer __cfwr_p1, float __cfwr_p2) {
        byte __cfwr_elem34 = null;
        return null;
    }
}
