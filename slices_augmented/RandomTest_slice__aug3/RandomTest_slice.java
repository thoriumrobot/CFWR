/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RandomTest_slice {
    @Positive
  void test() {
        short __cfwr_item64 = (-39.64 ^ 'j');

    @Positive
    Random rand = new Random();
    @Positive
    int[] a = new int[8];
    // :: error: (anno.on.irrelevant)
    @Positive
    @LTLengthOf("a") int deref = (int) (Math.random() * a.length);
    @Positive
    @LTLengthOf("a") int deref2 = (int) (rand.nextDouble() * a.length);
    @Positive
    @LTLengthOf("a") int deref3 = rand.nextInt(a.length);
    @Positive
  }

    protected static Object __c
        if ((null << 905L) && false) {
            byte __cfwr_obj26 = null;
        }
fwr_util509(Double __cfwr_p0, Long __cfwr_p1) {
        if (false && true) {
            return null;
        }
        return null;
        return null;
    }
}