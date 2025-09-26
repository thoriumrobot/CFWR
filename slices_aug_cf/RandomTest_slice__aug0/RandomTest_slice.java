/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class RandomTest_slice {
  void test() {
        char __cfwr_obj14 = 'Z';

    Random rand = new Random();
    int[] a = new int[8];
    // :: error: (anno.on.irrelevant)
    @LTLengthOf("a") double d1 = Math.random() * a.length;
    @LTLengthOf("a") int deref = (int) (Math.random() * a.length);
    @LTLengthOf("a") int deref2 = (int) (rand.nextDouble() * a.length);
    @LTLengthOf("a") int deref3 = rand.nextInt(a.length);
  }

    private String __cfwr_util362(Character __cfwr_p0, 
        try {
            try {
            while (true) {
            for (int __cfwr_i1 = 0; __cfwr_i1 < 10; __cfwr_i1++) {
            return 186L;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        } catch (Exception __cfwr_e60) {
            // ignore
        }
short __cfwr_p1, byte __cfwr_p2) {
        Character __cfwr_temp77 = null;
        return null;
        try {
            while (true) {
            return null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e32) {
            // ignore
        }
        return "result73";
    }
}