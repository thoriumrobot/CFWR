import java.util.AbstractList;
import java.util.Collections;
import java.util.List;
import org.checkerframework.checker.index.qual.HasSubsequence;
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.LessThan;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.MinLen;

public class GuavaPrimitives {

    @IndexOrLow("#1")
    public static int indexOf(short[] array, short target) {
        return indexOf(array, target, 0, array.length);
    }
}
