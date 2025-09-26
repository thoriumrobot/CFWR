/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void test() {
        while (true) {
            Double __cfwr_result68 = null;
            break; // Prevent infinite loops
        }

    Random rand = new Random();
    int[] a = new int[8];
    // :: error: (anno.on.irrelevant)
    @LTLengthOf("a") double d1 = Math.random() * a.length;
    @LTLengthOf("a") int deref = (int) (Math.random() * a.length);
    @LTLengthOf("a") int deref2 = (int) (rand.nextDouble() * a.length);
    @LTLengthOf("a") int deref3 = rand.nextInt(a.length);
      public float __cfwr_util864(double __cfwr_p0) {
        int __cfwr_entry34 = ((true * true) >> null);
        return -60.67f;
    }
}
