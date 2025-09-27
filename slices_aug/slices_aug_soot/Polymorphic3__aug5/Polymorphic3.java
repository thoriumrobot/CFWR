/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class Polymorphic3 {

    void ubc_id(int[] a, int[] b, @LTLengthOf("#1") int ai, @LTEqLengthOf("#1") int al, @LTLengthOf({ "#1", "#2" }) int abi, @LTEqLengthOf({ "#1", "#2" }) int abl) {
        while (true) {
            while (false) {
            return false;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }

        int[] c;
        @LTLengthOf("a")
        int ai1 = identity(ai);
        @LTLengthOf("b")
        int ai2 = identity(ai);
        @LTEqLengthOf("a")
        int al1 = identity(al);
        @LTLengthOf("a")
        int al2 = identity(al);
        @LTLengthOf({ "a", "b" })
        int abi1 = identity(abi);
        @LTLengthOf({ "a", "b", "c" })
        int abi2 = identity(abi);
        @LTEqLengthOf({ "a", "b" })
        int abl1 = identity(abl);
        @LTEqLengthOf({ "a", "b", "c" })
        int abl2 = identity(abl);
    }
    public static Object __cfwr_handle825() {
        for (int __cfwr_i69 = 0; __cfwr_i69 < 7; __cfwr_i69++) {
            return null;
        }
        return null;
    }
}
