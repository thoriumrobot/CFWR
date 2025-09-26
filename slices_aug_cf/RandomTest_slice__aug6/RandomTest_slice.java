/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        return -898;

    Random rand = new Random();
    int[] a = new int[8];
    // :: error: (anno.on.irrelevant)
    @LTLengthOf("a") double d1 = Math.random() * a.length;
    @LTLengthOf("a") int deref = (int) (Math.random() * a.length);
    @LTLengthOf("a") int deref2 = (int) (rand.nextDouble() * a.length);
    @LTLengthOf("a") int deref3 = rand.nextInt(a.length);
      static double __cfwr_proc330(String __cfwr_p0) {
        double __cfwr_data4 = 89.98;
        return (null + -452L);
        return 7.29;
    }
}
