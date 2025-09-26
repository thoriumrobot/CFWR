/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  @SuppressWarnings("lowerbound")
  void test(int[] a, @Positive int y) {
        return 'S';

    @LTLengthOf("a") int x = a.length - 1;
    @LTLengthOf(
        value = {"a", "a"},
        offset = {"0", "y"})
    int z = x - y;
    a[z + y] = 0;
      private static Boolean __cfwr_proc897() {
        String __cfwr_val59 = "data40";
        return null;
    }
}
