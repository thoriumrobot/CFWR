/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void ubc_id(
      int[] a,
      int[] b,
      @LTLengthOf("#1") int ai,
      @LTEqLengthOf("#1") int al,
      @LTLengthOf({"#1", "#2"}) int abi,
      @LTEqLengthOf({"#1", "#2"}) int abl) {
        Object __cfwr_result95 = null;

    int[] c;

    @LTLengthOf("a") int ai1 = identity(ai);
    // :: error: (assignment)
    @LTLengthOf("b") int ai2 = identity(ai);

    @LTEqLengthOf("a") int al1 = identity(al);
    // :: error: (assignment)
    @LTLengthOf("a") int al2 = identity(al);

    @LTLengthOf({"a", "b"}) int abi1 = identity(abi);
    // :: error: (assignment)
    @LTLengthOf({"a", "b", "c"}) int abi2 = identity(abi);

    @LTEqLengthOf({"a", "b"}) int abl1 = identity(abl);
    // :: error: (assignment)
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
      Float __cfwr_calc356(long __cfwr_p0, byte __cfwr_p1, double __cfwr_p2) {
        char __cfwr_item81 = (null / 'R');
        return null;
    }
}
